#!/usr/bin/env python3
"""Extract text from documents inside zip files under a path.

Usage:
  python extract_texts.py /path/to/input /path/to/output
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

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


def iter_zip_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.zip"):
        if path.is_file():
            yield path


def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1", errors="replace")


def extract_pdf_text(path: Path) -> Optional[str]:
    try:
        import PyPDF2  # type: ignore
    except ImportError:
        PyPDF2 = None

    if PyPDF2 is not None:
        try:
            reader = PyPDF2.PdfReader(str(path))
            parts = []
            for page in reader.pages:
                parts.append(page.extract_text() or "")
            text = "\n".join(parts).strip()
            if text:
                return text
        except Exception:
            pass

    try:
        import pdfplumber  # type: ignore
    except ImportError:
        pdfplumber = None

    if pdfplumber is not None:
        try:
            parts = []
            with pdfplumber.open(str(path)) as pdf:
                for page in pdf.pages:
                    parts.append(page.extract_text() or "")
            text = "\n".join(parts).strip()
            if text:
                return text
        except Exception:
            pass

    try:
        from pdfminer.high_level import extract_text  # type: ignore
    except ImportError:
        return None

    try:
        text = extract_text(str(path)).strip()
        return text or None
    except Exception:
        return None


def extract_docx_text(path: Path) -> Optional[str]:
    try:
        import docx  # type: ignore
    except ImportError:
        docx = None

    if docx is not None:
        try:
            document = docx.Document(str(path))
            return "\n".join(p.text for p in document.paragraphs).strip()
        except Exception:
            return None

    try:
        with zipfile.ZipFile(path) as docx_zip:
            xml_bytes = docx_zip.read("word/document.xml")
    except Exception:
        return None

    try:
        from xml.etree import ElementTree

        root = ElementTree.fromstring(xml_bytes)
        texts = []
        for node in root.iter():
            if node.tag.endswith("}t") and node.text:
                texts.append(node.text)
        return "".join(texts).strip()
    except Exception:
        return None


def extract_image_text(path: Path) -> Optional[str]:
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        return None

    try:
        import pytesseract  # type: ignore
    except ImportError:
        return None

    try:
        with Image.open(path) as image:
            return pytesseract.image_to_string(image).strip()
    except Exception:
        return None


def sanitize_filename(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return sanitized.strip("_") or "document"


def extract_text_from_file(path: Path) -> Optional[str]:
    ext = path.suffix.lower()
    if ext in TEXT_EXTENSIONS:
        return read_text_file(path)
    if ext == ".pdf":
        return extract_pdf_text(path)
    if ext == ".docx":
        return extract_docx_text(path)
    if ext in IMAGE_EXTENSIONS:
        return extract_image_text(path)
    return None


def process_zip(zip_path: Path, output_dir: Path) -> list[ExtractedText]:
    extracted: list[ExtractedText] = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(tmp_dir)

        for file_path in Path(tmp_dir).rglob("*"):
            if not file_path.is_file():
                continue
            text = extract_text_from_file(file_path)
            if text is None:
                continue
            extracted.append(
                ExtractedText(
                    source_zip=str(zip_path),
                    source_file=str(file_path.relative_to(tmp_dir)),
                    text=text,
                )
            )

    if extracted:
        zip_output_dir = output_dir / sanitize_filename(zip_path.stem)
        zip_output_dir.mkdir(parents=True, exist_ok=True)
        for item in extracted:
            output_name = sanitize_filename(Path(item.source_file).stem) + ".txt"
            (zip_output_dir / output_name).write_text(item.text, encoding="utf-8")

    return extracted


def write_jsonl(records: Iterable[ExtractedText], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    {
                        "source_zip": record.source_zip,
                        "source_file": record.source_file,
                        "text": record.text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract text from documents inside zip files."
    )
    parser.add_argument("input_path", help="Path containing zip files")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="extracted_text",
        help="Directory to write extracted text",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_path.exists():
        print(f"Input path not found: {input_path}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[ExtractedText] = []
    for zip_path in iter_zip_files(input_path):
        try:
            records = process_zip(zip_path, output_dir)
        except zipfile.BadZipFile:
            print(f"Skipping invalid zip: {zip_path}", file=sys.stderr)
            continue
        all_records.extend(records)

    jsonl_path = output_dir / "extracted_text.jsonl"
    write_jsonl(all_records, jsonl_path)

    print(f"Processed {len(all_records)} documents.")
    print(f"JSONL output: {jsonl_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
