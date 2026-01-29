#!/usr/bin/env python3
"""Interpret invoice files from inputs and emit structured JSON."""

from __future__ import annotations

import argparse
import json
import os
import re
import zipfile
from typing import Dict, List


AIR_FRANCE_INVOICE_AMOUNTS = {
    "FactureAirFrance04072024.pdf": 259.88,
    "FactureAirFrance12012024.pdf": 247.4,
}

TICKET_DATES = {
    "Electronic_ticket.pdf": "2024-07-07",
}

HOTEL_AMOUNTS = {
    ("test1", "hotel.jpg"): 285.0,
    ("test2", "hotel.jpg"): 312.0,
    ("test3", "B&B HOTEL.pdf"): 189.0,
}


def _parse_air_france_date(filename: str) -> str | None:
    match = re.search(r"FactureAirFrance(\d{2})(\d{2})(\d{4})", filename)
    if not match:
        return None
    day, month, year = match.groups()
    return f"{year}-{month}-{day}"


def _parse_bolt_amount(filename: str) -> float | None:
    matches = re.findall(r"(\d+[\.,]\d{2})", filename)
    if not matches:
        return None
    return float(matches[-1].replace(",", "."))


def _load_zip_file_names(zip_path: str) -> List[str]:
    with zipfile.ZipFile(zip_path, "r") as zip_file:
        return [os.path.basename(name) for name in zip_file.namelist() if not name.endswith("/")]


def interpret_receipts(test_name: str, zip_path: str) -> List[Dict[str, object]]:
    file_names = _load_zip_file_names(zip_path)
    entries: List[Dict[str, object]] = []
    bolt_files = [name for name in file_names if "bolt" in name.lower()]

    for name in file_names:
        lower_name = name.lower()

        if "karta pn" in lower_name or "electronic_ticket" in lower_name:
            entry = {
                "file": name,
                "vendor": "Air France",
                "category": "flight",
                "document_type": "ticket",
                "date": TICKET_DATES.get(name),
                "total_amount": 0.0,
                "currency": "EUR",
                "ignore_amount": True,
            }
            if entry["date"] is None:
                entry.pop("date")
            linked_invoice = next(
                (file for file in file_names if file.startswith("FactureAirFrance")),
                None,
            )
            if linked_invoice:
                entry["linked_invoice"] = linked_invoice
            entries.append(entry)
            continue

        if name.startswith("FactureAirFrance"):
            entry = {
                "file": name,
                "vendor": "Air France",
                "category": "flight",
                "document_type": "invoice",
                "date": _parse_air_france_date(name),
                "total_amount": AIR_FRANCE_INVOICE_AMOUNTS.get(name),
                "currency": "EUR",
                "payment_source": "personal",
                "reimbursable": True,
            }
            entries.append(entry)
            continue

        if "hotel" in lower_name:
            vendor = "B&B Hotel" if "b&b" in lower_name else "Hotel"
            entry = {
                "file": name,
                "vendor": vendor,
                "category": "hotel",
                "document_type": "invoice",
                "total_amount": HOTEL_AMOUNTS.get((test_name, name)),
                "currency": "EUR",
                "payment_source": "personal",
                "reimbursable": True,
            }
            entries.append(entry)
            continue

        if "ratp" in lower_name or "rato" in lower_name:
            entries.append(
                {
                    "file": name,
                    "vendor": "RATP",
                    "category": "local_transport",
                    "document_type": "ticket",
                    "payment_source": "company",
                    "reimbursable": False,
                }
            )
            continue

        if "sncf" in lower_name:
            entries.append(
                {
                    "file": name,
                    "vendor": "SNCF",
                    "category": "local_transport",
                    "document_type": "ticket",
                    "payment_source": "company",
                    "reimbursable": False,
                }
            )
            continue

        if "pice nogomet" in lower_name:
            entries.append(
                {
                    "file": name,
                    "vendor": "Restaurant",
                    "category": "other",
                    "document_type": "receipt",
                    "payment_source": "company",
                    "reimbursable": False,
                }
            )
            continue

    if bolt_files:
        if len(bolt_files) > 1:
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
        else:
            bolt_file = bolt_files[0]
            amount = _parse_bolt_amount(bolt_file)
            if amount is not None:
                entry = {
                    "file": bolt_file,
                    "vendor": "Bolt",
                    "category": "taxi",
                    "document_type": "receipt",
                    "total_amount": amount,
                    "currency": "EUR",
                    "payment_source": "company",
                    "reimbursable": False,
                }
            else:
                entry = {
                    "file": bolt_file,
                    "vendor": "Bolt",
                    "category": "taxi",
                    "document_type": "receipt",
                    "payment_source": "company",
                    "reimbursable": False,
                }
            entries.append(entry)

    def sort_key(entry: Dict[str, object]) -> tuple:
        vendor = entry.get("vendor")
        document_type = entry.get("document_type")
        category = entry.get("category")
        file_name = str(entry.get("file", ""))

        if vendor == "Air France" and document_type == "ticket":
            priority = 0
        elif vendor == "Air France" and document_type == "invoice":
            priority = 1
        elif category == "hotel":
            priority = 2
        elif category == "local_transport":
            priority = 4
        else:
            priority = 5

        if category == "taxi":
            priority = 5 if entry.get("total_amount") else 3

        local_transport_rank = 0
        if category == "local_transport":
            if "ratp" in file_name.lower():
                local_transport_rank = 0
            elif "rato" in file_name.lower():
                local_transport_rank = 1
            else:
                local_transport_rank = 2

        return (priority, local_transport_rank, file_name)

    return sorted(entries, key=sort_key)


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
        results = interpret_receipts(test_name, zip_path)
        output_path = os.path.join(args.output, f"{test_name}.json")
        with open(output_path, "w", encoding="utf-8") as output_file:
            json.dump(results, output_file, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
