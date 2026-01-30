#!/usr/bin/env python3
"""Extract text from documents inside zip files under a path.

IMPROVEMENTS:
- Better OCR with preprocessing (deskew, denoise, contrast)
- Multilingual OCR (English + French for your use case)
- Progress bars and better logging
- Parallel processing option
- Better error handling and reporting
- Image preprocessing for better quality
- Multiple OCR attempts with different settings

Usage:
  python extract_texts_improved.py /path/to/input /path/to/output
  python extract_texts_improved.py /path/to/input /path/to/output --parallel
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

# Try to import optional dependencies
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable

TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".csv",
    ".tsv",
    ".json",
    ".xml",
    ".html",
    ".htm",
    ".rtf",
    ".log",
}

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tiff",
    ".tif",
    ".bmp",
    ".gif",
    ".webp",
}


@dataclass
class ExtractedText:
    source_zip: str
    source_file: str
    text: str
    method: str = "unknown"  # Track which extraction method worked
    char_count: int = 0
    confidence: float = 0.0  # For OCR confidence


def iter_zip_files(root: Path) -> Iterable[Path]:
    """Find all zip files recursively."""
    for path in root.rglob("*.zip"):
        if path.is_file():
            yield path


def read_text_file(path: Path) -> Optional[str]:
    """Read plain text files with encoding fallback."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="latin-1", errors="replace")
        except Exception:
            return None


def preprocess_image_for_ocr(image):
    """Preprocess image for better OCR accuracy."""
    try:
        import cv2
        import numpy as np
        
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Increase contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
        
        # Binarization (Otsu's method)
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
        
    except ImportError:
        # If OpenCV not available, return original
        return image
    except Exception:
        # If preprocessing fails, return original
        return image


def extract_image_text_advanced(path: Path) -> Optional[tuple[str, float]]:
    """
    Extract text from images with preprocessing and multilingual support.
    Returns (text, confidence) tuple.
    """
    try:
        from PIL import Image
    except ImportError:
        return None

    try:
        import pytesseract
    except ImportError:
        return None

    try:
        with Image.open(path) as image:
            # Convert to RGB if needed
            if image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')
            
            # Try multiple OCR strategies
            results = []
            
            # Strategy 1: Direct OCR with multilingual (English + French)
            try:
                text1 = pytesseract.image_to_string(image, lang='eng+fra').strip()
                if text1:
                    # Get confidence if available
                    try:
                        data = pytesseract.image_to_data(image, lang='eng+fra', output_type=pytesseract.Output.DICT)
                        confidences = [int(conf) for conf in data['conf'] if conf != '-1']
                        avg_conf = sum(confidences) / len(confidences) if confidences else 0
                        results.append((text1, avg_conf, "direct"))
                    except:
                        results.append((text1, 75.0, "direct"))
            except Exception as e:
                print(f"  [Warning] Direct OCR failed: {e}")
            
            # Strategy 2: With preprocessing
            try:
                preprocessed = preprocess_image_for_ocr(image)
                text2 = pytesseract.image_to_string(preprocessed, lang='eng+fra').strip()
                if text2:
                    results.append((text2, 80.0, "preprocessed"))
            except Exception as e:
                print(f"  [Warning] Preprocessed OCR failed: {e}")
            
            # Strategy 3: Upscale small images
            try:
                if image.width < 1000 or image.height < 1000:
                    scale_factor = max(1000 / image.width, 1000 / image.height)
                    new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
                    upscaled = image.resize(new_size, Image.Resampling.LANCZOS)
                    text3 = pytesseract.image_to_string(upscaled, lang='eng+fra').strip()
                    if text3:
                        results.append((text3, 70.0, "upscaled"))
            except Exception as e:
                print(f"  [Warning] Upscaled OCR failed: {e}")
            
            # Pick the best result (longest text with decent confidence)
            if results:
                best = max(results, key=lambda x: len(x[0]) * (x[1] / 100))
                return (best[0], best[1])
            
            return None
            
    except Exception as e:
        print(f"  [Error] Image OCR failed for {path}: {e}")
        return None


def extract_pdf_text(path: Path) -> Optional[tuple[str, str]]:
    """
    Extract text from PDF with fallback chain.
    Returns (text, method) tuple.
    """
    # Try pdfplumber first (best for tables and formatting)
    try:
        import pdfplumber
        parts = []
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                parts.append(text)
        text = "\n".join(parts).strip()
        if text and len(text) > 20:  # Minimum threshold
            return (text, "pdfplumber")
    except ImportError:
        pass
    except Exception as e:
        print(f"  [Warning] pdfplumber failed: {e}")

    # Try PyPDF2
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(str(path))
        parts = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        text = "\n".join(parts).strip()
        if text and len(text) > 20:
            return (text, "PyPDF2")
    except ImportError:
        pass
    except Exception as e:
        print(f"  [Warning] PyPDF2 failed: {e}")

    # Try pdfminer
    try:
        from pdfminer.high_level import extract_text
        text = extract_text(str(path)).strip()
        if text and len(text) > 20:
            return (text, "pdfminer")
    except ImportError:
        pass
    except Exception as e:
        print(f"  [Warning] pdfminer failed: {e}")

    # Last resort: OCR the PDF
    try:
        import pdf2image
        import pytesseract
        
        print(f"  [Info] Attempting OCR on PDF (may be slow)...")
        images = pdf2image.convert_from_path(str(path), dpi=300)
        parts = []
        for i, image in enumerate(images):
            try:
                text = pytesseract.image_to_string(image, lang='eng+fra').strip()
                if text:
                    parts.append(text)
            except Exception as e:
                print(f"  [Warning] OCR failed on page {i+1}: {e}")
        
        text = "\n\n".join(parts).strip()
        if text:
            return (text, "pdf_ocr")
    except ImportError:
        pass
    except Exception as e:
        print(f"  [Warning] PDF OCR failed: {e}")

    return None


def extract_docx_text(path: Path) -> Optional[str]:
    """Extract text from DOCX files."""
    try:
        import docx
        document = docx.Document(str(path))
        # Include both paragraphs and tables
        parts = []
        for paragraph in document.paragraphs:
            if paragraph.text.strip():
                parts.append(paragraph.text)
        for table in document.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text.strip():
                        parts.append(cell.text)
        return "\n".join(parts).strip()
    except ImportError:
        pass
    except Exception:
        pass

    # Fallback: manual XML extraction
    try:
        with zipfile.ZipFile(path) as docx_zip:
            xml_bytes = docx_zip.read("word/document.xml")
        from xml.etree import ElementTree
        root = ElementTree.fromstring(xml_bytes)
        texts = []
        for node in root.iter():
            if node.tag.endswith("}t") and node.text:
                texts.append(node.text)
        return " ".join(texts).strip()
    except Exception:
        return None


def sanitize_filename(name: str) -> str:
    """Sanitize filename for safe filesystem usage."""
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return sanitized.strip("_") or "document"


def extract_text_from_file(path: Path) -> Optional[ExtractedText]:
    """
    Extract text from a file using appropriate method.
    Returns ExtractedText with metadata.
    """
    ext = path.suffix.lower()
    text = None
    method = "unknown"
    confidence = 0.0
    
    # Plain text files
    if ext in TEXT_EXTENSIONS:
        text = read_text_file(path)
        method = "plaintext"
        confidence = 100.0
    
    # PDF files
    elif ext == ".pdf":
        result = extract_pdf_text(path)
        if result:
            text, method = result
            confidence = 95.0 if method != "pdf_ocr" else 75.0
    
    # Word documents
    elif ext == ".docx":
        text = extract_docx_text(path)
        method = "docx"
        confidence = 95.0
    
    # Image files (OCR)
    elif ext in IMAGE_EXTENSIONS:
        result = extract_image_text_advanced(path)
        if result:
            text, confidence = result
            method = "ocr"
    
    if text:
        return ExtractedText(
            source_zip="",  # Will be filled later
            source_file=str(path),
            text=text,
            method=method,
            char_count=len(text),
            confidence=confidence
        )
    
    return None


def process_zip(zip_path: Path, output_dir: Path, verbose: bool = True) -> list[ExtractedText]:
    """Process a single zip file and extract all text content."""
    extracted: list[ExtractedText] = []
    
    if verbose:
        print(f"\n📦 Processing: {zip_path.name}")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(tmp_dir)
        except zipfile.BadZipFile:
            print(f"  ❌ Invalid zip file: {zip_path}")
            return []

        files = list(Path(tmp_dir).rglob("*"))
        if verbose:
            print(f"  Found {len([f for f in files if f.is_file()])} files")
        
        iterator = tqdm(files, desc="  Extracting", leave=False) if HAS_TQDM and verbose else files
        
        for file_path in iterator:
            if not file_path.is_file():
                continue
            
            result = extract_text_from_file(file_path)
            if result is None:
                continue
            
            # Update source_zip and relative path
            result.source_zip = str(zip_path)
            result.source_file = str(file_path.relative_to(tmp_dir))
            
            extracted.append(result)
            
            if verbose and not HAS_TQDM:
                print(f"    ✓ {result.source_file} ({result.char_count} chars, {result.method})")

    # Write individual text files
    if extracted:
        zip_output_dir = output_dir / sanitize_filename(zip_path.stem)
        zip_output_dir.mkdir(parents=True, exist_ok=True)
        for item in extracted:
            output_name = sanitize_filename(Path(item.source_file).stem) + ".txt"
            (zip_output_dir / output_name).write_text(item.text, encoding="utf-8")
    
    if verbose:
        print(f"  ✅ Extracted {len(extracted)} documents")
    
    return extracted


def write_jsonl(records: Iterable[ExtractedText], output_path: Path) -> None:
    """Write extracted text records to JSONL format."""
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    {
                        "source_zip": record.source_zip,
                        "source_file": record.source_file,
                        "text": record.text,
                        "method": record.method,
                        "char_count": record.char_count,
                        "confidence": record.confidence,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def generate_summary_report(records: list[ExtractedText], output_dir: Path) -> None:
    """Generate a summary report of the extraction."""
    from collections import Counter
    
    total_docs = len(records)
    total_chars = sum(r.char_count for r in records)
    methods = Counter(r.method for r in records)
    avg_confidence = sum(r.confidence for r in records) / total_docs if total_docs > 0 else 0
    
    report = f"""
# Extraction Summary Report

## Overall Statistics
- Total documents processed: {total_docs}
- Total characters extracted: {total_chars:,}
- Average confidence: {avg_confidence:.1f}%

## Extraction Methods
"""
    for method, count in methods.most_common():
        percentage = (count / total_docs * 100) if total_docs > 0 else 0
        report += f"- {method}: {count} documents ({percentage:.1f}%)\n"
    
    report += "\n## Documents by Confidence\n"
    high_conf = sum(1 for r in records if r.confidence >= 90)
    med_conf = sum(1 for r in records if 70 <= r.confidence < 90)
    low_conf = sum(1 for r in records if r.confidence < 70)
    
    report += f"- High (≥90%): {high_conf}\n"
    report += f"- Medium (70-89%): {med_conf}\n"
    report += f"- Low (<70%): {low_conf}\n"
    
    # Documents that might need review
    report += "\n## Documents Requiring Review (low confidence or OCR)\n"
    review_docs = [r for r in records if r.confidence < 80 or 'ocr' in r.method]
    for doc in review_docs[:20]:  # Show first 20
        report += f"- {doc.source_file} ({doc.method}, {doc.confidence:.0f}% conf, {doc.char_count} chars)\n"
    
    if len(review_docs) > 20:
        report += f"... and {len(review_docs) - 20} more\n"
    
    report_path = output_dir / "extraction_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n📊 Summary report: {report_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract text from documents inside zip files with advanced OCR."
    )
    parser.add_argument("input_path", help="Path containing zip files")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="extracted_text",
        help="Directory to write extracted text",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Process zip files in parallel (faster but uses more memory)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output"
    )
    args = parser.parse_args()

    input_path = Path(args.input_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_path.exists():
        print(f"❌ Input path not found: {input_path}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 Starting text extraction...")
    print(f"📂 Input: {input_path}")
    print(f"📂 Output: {output_dir}")

    zip_files = list(iter_zip_files(input_path))
    if not zip_files:
        print("❌ No zip files found", file=sys.stderr)
        return 1
    
    print(f"📦 Found {len(zip_files)} zip file(s)")

    all_records: list[ExtractedText] = []
    
    if args.parallel:
        print(f"⚡ Using {args.workers} parallel workers")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_zip, zp, output_dir, not args.quiet): zp 
                for zp in zip_files
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing zips") if HAS_TQDM else as_completed(futures):
                try:
                    records = future.result()
                    all_records.extend(records)
                except Exception as e:
                    zip_path = futures[future]
                    print(f"❌ Error processing {zip_path}: {e}", file=sys.stderr)
    else:
        for zip_path in zip_files:
            try:
                records = process_zip(zip_path, output_dir, not args.quiet)
                all_records.extend(records)
            except Exception as e:
                print(f"❌ Error processing {zip_path}: {e}", file=sys.stderr)

    # Write JSONL output
    jsonl_path = output_dir / "extracted_text.jsonl"
    write_jsonl(all_records, jsonl_path)
    
    # Generate summary report
    if not args.quiet:
        generate_summary_report(all_records, output_dir)

    print(f"\n✅ Done! Processed {len(all_records)} documents")
    print(f"📄 JSONL output: {jsonl_path}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
