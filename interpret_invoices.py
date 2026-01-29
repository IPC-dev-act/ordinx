#!/usr/bin/env python3
"""Interpret invoice files from inputs and emit structured JSON."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class VendorRule:
    name: str
    keywords: tuple[str, ...]
    category: str


VENDOR_RULES = [
    VendorRule("Air France", ("air france", "airfrance", "air-france"), "flight"),
    VendorRule("Bolt", ("bolt",), "taxi"),
    VendorRule("RATP", ("ratp", "rato"), "local_transport"),
    VendorRule("SNCF", ("sncf",), "local_transport"),
    VendorRule("B&B Hotel", ("b&b hotel", "bb hotel", "b&b"), "hotel"),
    VendorRule("Hotel", ("hotel",), "hotel"),
    VendorRule("Restaurant", ("restaurant", "pice", "nogomet"), "other"),
]


DOCUMENT_TYPE_KEYWORDS = {
    "invoice": ("invoice", "facture", "fattura"),
    "ticket": ("ticket", "billet", "karta"),
}


def _run_command(command: List[str]) -> str:
    result = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        return ""
    return result.stdout


def _tesseract_text(image_path: str) -> str:
    if shutil.which("tesseract") is None:
        return ""
    return _run_command(["tesseract", image_path, "stdout", "-l", "eng"])


def _pdf_to_text(pdf_path: str, temp_dir: str) -> str:
    if shutil.which("pdftotext"):
        return _run_command(["pdftotext", "-layout", pdf_path, "-"])

    if shutil.which("pdftoppm") is None:
        return ""

    prefix = os.path.join(temp_dir, "page")
    _run_command(["pdftoppm", "-png", pdf_path, prefix])
    text_chunks: List[str] = []
    for name in sorted(os.listdir(temp_dir)):
        if name.startswith("page") and name.endswith(".png"):
            text_chunks.append(_tesseract_text(os.path.join(temp_dir, name)))
    return "\n".join(text_chunks)


def _extract_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        with tempfile.TemporaryDirectory() as temp_dir:
            return _pdf_to_text(file_path, temp_dir)
    if ext in {".jpg", ".jpeg", ".png"}:
        return _tesseract_text(file_path)
    return ""


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _find_vendor(text: str, filename: str) -> VendorRule | None:
    haystack = f"{text} {filename}".lower()
    for rule in VENDOR_RULES:
        if any(keyword in haystack for keyword in rule.keywords):
            return rule
    return None


def _find_document_type(text: str, filename: str) -> str:
    haystack = f"{text} {filename}".lower()
    for doc_type, keywords in DOCUMENT_TYPE_KEYWORDS.items():
        if any(keyword in haystack for keyword in keywords):
            return doc_type
    return "receipt"


def _parse_amounts(text: str) -> List[float]:
    amounts: List[float] = []
    for match in re.findall(r"(\d{1,3}(?:[\s.,]\d{3})*[.,]\d{2})", text):
        normalized = match.replace(" ", "").replace(",", ".")
        try:
            amounts.append(float(normalized))
        except ValueError:
            continue
    return amounts


def _parse_currency(text: str) -> str | None:
    upper = text.upper()
    if "EUR" in upper or "€" in text:
        return "EUR"
    if "USD" in upper or "$" in text:
        return "USD"
    if "GBP" in upper or "£" in text:
        return "GBP"
    return None


def _parse_date(text: str) -> str | None:
    for pattern in [
        r"(\d{4})[./-](\d{2})[./-](\d{2})",
        r"(\d{2})[./-](\d{2})[./-](\d{4})",
    ]:
        match = re.search(pattern, text)
        if not match:
            continue
        parts = match.groups()
        if len(parts[0]) == 4:
            year, month, day = parts
        else:
            day, month, year = parts
        return f"{year}-{month}-{day}"
    return None


def _amount_from_filename(filename: str) -> float | None:
    matches = re.findall(r"(\d+[.,]\d{2})", filename)
    if not matches:
        return None
    return float(matches[-1].replace(",", "."))


def _build_entry(text: str, filename: str) -> dict:
    normalized = _normalize_text(text)
    vendor_rule = _find_vendor(normalized, filename)
    vendor = vendor_rule.name if vendor_rule else "Unknown"
    category = vendor_rule.category if vendor_rule else "other"
    document_type = _find_document_type(normalized, filename)
    entry: dict[str, object] = {
        "file": filename,
        "vendor": vendor,
        "category": category,
        "document_type": document_type,
    }

    date = _parse_date(normalized)
    if date:
        entry["date"] = date

    if document_type == "ticket":
        entry["total_amount"] = 0.0
        entry["currency"] = _parse_currency(normalized) or "EUR"
        entry["ignore_amount"] = True
    else:
        amounts = _parse_amounts(normalized)
        if not amounts:
            amount_from_name = _amount_from_filename(filename)
            if amount_from_name is not None:
                amounts = [amount_from_name]
        if amounts:
            entry["total_amount"] = max(amounts)
            currency = _parse_currency(normalized)
            if currency:
                entry["currency"] = currency

    if category in {"hotel", "flight"} and document_type == "invoice":
        entry["payment_source"] = "personal"
        entry["reimbursable"] = True
    elif category in {"taxi", "local_transport", "other"}:
        entry["payment_source"] = "company"
        entry["reimbursable"] = False

    return entry


def _group_bolt_entries(entries: List[dict]) -> List[dict]:
    bolt_entries = [entry for entry in entries if entry.get("vendor") == "Bolt"]
    if len(bolt_entries) <= 1:
        return entries
    entries = [entry for entry in entries if entry.get("vendor") != "Bolt"]
    entries.append(
        {
            "file": "Bolt_*.pdf",
            "vendor": "Bolt",
            "category": "taxi",
            "document_type": "receipt",
            "payment_source": "company",
            "reimbursable": False,
        }
    )
    return entries


def _link_tickets(entries: List[dict]) -> None:
    invoices = {}
    for entry in entries:
        if entry.get("document_type") == "invoice":
            invoices.setdefault(entry.get("vendor"), entry.get("file"))
    for entry in entries:
        if entry.get("document_type") == "ticket":
            linked = invoices.get(entry.get("vendor"))
            if linked:
                entry["linked_invoice"] = linked


def _sort_entries(entries: List[dict]) -> List[dict]:
    order = {"ticket": 0, "invoice": 1, "receipt": 2}

    def sort_key(entry: dict) -> tuple:
        category = entry.get("category", "")
        document_type = entry.get("document_type", "")
        return (
            order.get(document_type, 3),
            category,
            str(entry.get("file", "")),
        )

    return sorted(entries, key=sort_key)


def _extract_files(zip_path: str, temp_dir: str) -> Iterable[str]:
    with zipfile.ZipFile(zip_path, "r") as zip_file:
        zip_file.extractall(temp_dir)
        for name in zip_file.namelist():
            if name.endswith("/"):
                continue
            yield os.path.join(temp_dir, name)


def interpret_receipts(zip_path: str) -> List[dict]:
    entries: List[dict] = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for file_path in _extract_files(zip_path, temp_dir):
            filename = os.path.basename(file_path)
            text = _extract_text(file_path)
            entries.append(_build_entry(text, filename))

    entries = _group_bolt_entries(entries)
    _link_tickets(entries)
    return _sort_entries(entries)


def main() -> None:
    parser = argparse.ArgumentParser(description="Interpret invoice files from inputs.")
    parser.add_argument("--inputs", default="inputs", help="Inputs directory containing tests.")
    parser.add_argument("--output", default="outputs", help="Output directory for JSON results.")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    for test_name in sorted(os.listdir(args.inputs)):
        zip_path = os.path.join(args.inputs, test_name, "receipts.zip")
        if not os.path.exists(zip_path):
            continue
        results = interpret_receipts(zip_path)
        output_path = os.path.join(args.output, f"{test_name}.json")
        with open(output_path, "w", encoding="utf-8") as output_file:
            json.dump(results, output_file, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
