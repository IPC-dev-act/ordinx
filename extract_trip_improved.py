#!/usr/bin/env python3
"""
IMPROVED: Extract travel + expense info from OCR'd documents stored in a JSONL file.

Key improvements:
1. Better date parsing (handles DD/MM/YYYY, DD.MM.YYYY, DDMONTHYYYY formats)
2. Enhanced prompt with specific examples from your data
3. Better handling of short/noisy OCR text
4. More flexible JSON extraction
5. Currency detection improvements
6. Better cost extraction for various document types
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


# ----------------------------
# Date normalization helpers
# ----------------------------
def normalize_date_to_iso(date_str: str) -> Optional[str]:
    """
    Convert various date formats to ISO YYYY-MM-DD.
    Handles: DD/MM/YYYY, DD.MM.YYYY, DDMONTHYYYY, DD MONTH YYYY
    """
    if not date_str or not isinstance(date_str, str):
        return None
    
    date_str = date_str.strip()
    
    # Already ISO format
    if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
        return date_str
    
    # Try DD/MM/YYYY or DD.MM.YYYY
    match = re.match(r'^(\d{1,2})[/\.](\d{1,2})[/\.](\d{4})$', date_str)
    if match:
        day, month, year = match.groups()
        try:
            dt = datetime(int(year), int(month), int(day))
            return dt.strftime('%Y-%m-%d')
        except ValueError:
            pass
    
    # Try DD MONTH YYYY (e.g., "02 Feb 2024")
    month_names = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
        'janvier': 1, 'février': 2, 'mars': 3, 'avril': 4, 'mai': 5, 'juin': 6,
        'juillet': 7, 'août': 8, 'septembre': 9, 'octobre': 10, 'novembre': 11, 'décembre': 12
    }
    
    match = re.match(r'^(\d{1,2})\s+([a-zé]+)\s+(\d{4})$', date_str.lower())
    if match:
        day, month_str, year = match.groups()
        month = month_names.get(month_str[:3])
        if month:
            try:
                dt = datetime(int(year), month, int(day))
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                pass
    
    return None


# ----------------------------
# Robust JSON extraction
# ----------------------------
def extract_last_valid_json(text: str) -> Dict[str, Any]:
    """
    Extract the LAST valid JSON object from model output.
    More robust extraction with fallback to finding JSON blocks.
    """
    # First try: find all {...} blocks
    candidates = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, flags=re.DOTALL)
    
    parsed: List[Dict[str, Any]] = []
    for c in candidates:
        try:
            obj = json.loads(c)
            # Only accept if it looks like our expected structure
            if isinstance(obj, dict):
                parsed.append(obj)
        except json.JSONDecodeError:
            continue
    
    if not parsed:
        # Second try: look for JSON markdown blocks
        md_json = re.findall(r'```json\s*(\{.*?\})\s*```', text, flags=re.DOTALL)
        for c in md_json:
            try:
                parsed.append(json.loads(c))
            except json.JSONDecodeError:
                continue
    
    if not parsed:
        raise ValueError("No valid JSON found in model output.\n---OUTPUT---\n" + text[:1000])
    
    return parsed[-1]


def clamp_text(s: str, max_chars: int) -> str:
    """Smart text truncation that preserves key information."""
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    
    # For very short texts, don't truncate
    if len(s) < 200:
        return s
    
    # Keep more of the beginning (usually has headers, dates, key info)
    head = s[: int(max_chars * 0.85)]
    tail = s[-int(max_chars * 0.15) :]
    return head + "\n[... truncated ...]\n" + tail


# ----------------------------
# Enhanced LLM prompt
# ----------------------------
def build_messages(doc_text: str, source_file: str) -> List[Dict[str, str]]:
    """
    Enhanced prompt with better instructions and examples.
    """
    system = (
        "You are a precise data extraction assistant for travel and expense documents. "
        "Extract structured information from OCR'd text that may contain errors or formatting issues. "
        "Output ONLY valid JSON. No markdown formatting. No explanations outside the JSON."
    )

    user = f"""Extract travel and expense information from this document.

DOCUMENT: {source_file}

IMPORTANT INSTRUCTIONS:
1. **Dates**: Convert all dates to ISO format YYYY-MM-DD
   - "04/02/2024" → "2024-02-04"
   - "07-02-2024" → "2024-02-07"
   - "02 Feb 2024" → "2024-02-04"
   - For flight tickets: trip_start_date = departure date, trip_end_date = return date
   - For hotel stays: extract check-in and check-out dates

2. **Amounts**: Extract numeric values with decimals
   - "14,00" → "14.00"
   - "280.20" → "280.20"
   - Include currency code (EUR, USD, HRK, etc.)

3. **Document Types**:
   - Flight ticket/boarding pass → doc_type: "flight_ticket", category: "air_travel"
   - Hotel invoice/facture → doc_type: "hotel_invoice", category: "hotel"
   - Taxi/ride receipt (Bolt, Uber, RATP) → doc_type: "receipt", category: "local_transport"
   - Restaurant/meal receipt → doc_type: "receipt", category: "meal"

4. **Key Fields**:
   - destination: Extract city name (Zagreb, Paris, etc.)
   - vendor: Airline name (Air France), hotel name (B&B Hotel), or service (Bolt, RATP)
   - total_amount: The final amount charged
   - passenger_name: If present

5. **Handle OCR Errors**: Text may have formatting issues, missing spaces, or character recognition errors. Be flexible.

OUTPUT FORMAT (return exactly this JSON structure):
{{
  "source_file_summary": {{
    "doc_type": "flight_ticket|hotel_invoice|receipt|other",
    "category": "air_travel|hotel|meal|local_transport|other",
    "vendor": "vendor or company name",
    "country": "country name or code",
    "city": "city name"
  }},
  "trip": {{
    "trip_start_date": "YYYY-MM-DD or null",
    "trip_end_date": "YYYY-MM-DD or null",
    "origin": "origin city/airport",
    "destination": "destination city/airport"
  }},
  "costs": {{
    "currency": "EUR|USD|HRK|etc",
    "total_amount": "numeric string like 280.50",
    "air_travel_cost": "amount if flight",
    "hotel_cost": "amount if hotel",
    "local_transport_cost": "amount if taxi/metro",
    "meal_cost": "amount if meal",
    "tax_amount": "tax/VAT amount if shown"
  }},
  "passenger_info": {{
    "passenger_name": "name if present",
    "ticket_number": "ticket/booking number"
  }},
  "overall_description": "Brief summary of this expense",
  "evidence": [
    {{"field": "field_name", "quote": "relevant text snippet"}}
  ]
}}

DOCUMENT TEXT:
\"\"\"
{doc_text}
\"\"\"

Return only the JSON, no other text.
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ----------------------------
# Model loading + generation
# ----------------------------
def load_qwen_4bit(model_path: str):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    return tokenizer, model


@torch.inference_mode()
def run_extract(tokenizer, model, doc_text: str, source_file: str, max_new_tokens: int = 600) -> Dict[str, Any]:
    messages = build_messages(doc_text, source_file)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.1,  # More deterministic
        eos_token_id=tokenizer.eos_token_id,
    )

    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    
    # Debug: save raw output
    print(f"  [Debug] Raw output length: {len(decoded)} chars")
    
    try:
        data = extract_last_valid_json(decoded)
    except ValueError as e:
        print(f"  [Warning] JSON extraction failed: {e}")
        # Return minimal structure
        data = {
            "source_file_summary": {"doc_type": "unknown", "category": "other"},
            "trip": {},
            "costs": {},
            "passenger_info": {},
            "evidence": [{"field": "error", "quote": str(e)[:200]}]
        }

    # Normalize dates in the extracted data
    trip = data.get("trip", {})
    if trip.get("trip_start_date"):
        normalized = normalize_date_to_iso(trip["trip_start_date"])
        if normalized:
            trip["trip_start_date"] = normalized
    if trip.get("trip_end_date"):
        normalized = normalize_date_to_iso(trip["trip_end_date"])
        if normalized:
            trip["trip_end_date"] = normalized

    # Ensure structure
    data.setdefault("source_file_summary", {})
    data.setdefault("trip", {})
    data.setdefault("costs", {})
    data.setdefault("passenger_info", {})
    data.setdefault("evidence", [])
    
    return data


# ----------------------------
# Input reading
# ----------------------------
def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


# ----------------------------
# Aggregation helpers
# ----------------------------
def trip_key(d: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    t = (d.get("trip") or {})
    return (t.get("trip_start_date"), t.get("trip_end_date"), t.get("destination"))


def to_float(x: Any) -> Optional[float]:
    """Convert string/number to float, handling various formats."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if not s:
        return None
    # Remove spaces, convert comma to dot
    s = s.replace(" ", "").replace(",", ".")
    # Remove currency symbols
    s = re.sub(r'[€$£¥₹]', '', s)
    try:
        return float(s)
    except ValueError:
        return None


def safe_get(d: Dict[str, Any], *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to extracted_text.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct", help="Model name or path")
    ap.add_argument("--outdir", default="out", help="Output directory")
    ap.add_argument("--max-chars", type=int, default=8000, help="Max chars per doc")
    ap.add_argument("--max-new-tokens", type=int, default=600, help="Max new tokens")
    ap.add_argument("--debug", action="store_true", help="Save debug outputs")
    args = ap.parse_args()

    in_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"📥 Reading: {in_path}")
    docs = read_jsonl(in_path)
    print(f"📄 Documents: {len(docs)}")

    print(f"🧠 Loading model: {args.model}")
    tokenizer, model = load_qwen_4bit(args.model)
    print("✅ Model loaded")

    extracted: List[Dict[str, Any]] = []
    
    for i, row in enumerate(docs, 1):
        src_zip = row.get("source_zip")
        src_file = row.get("source_file")
        text = clamp_text(row.get("text", ""), args.max_chars)

        print(f"\n[{i}/{len(docs)}] Processing: {src_file}")
        print(f"  Text length: {len(row.get('text', ''))} chars")
        
        if args.debug:
            debug_file = outdir / f"debug_{i}_input.txt"
            debug_file.write_text(text, encoding="utf-8")
        
        try:
            data = run_extract(tokenizer, model, text, src_file, max_new_tokens=args.max_new_tokens)
            print(f"  ✅ Extracted: {data.get('source_file_summary', {}).get('doc_type', 'unknown')}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
            data = {
                "source_file_summary": {"doc_type": "error", "category": "error"},
                "trip": {},
                "costs": {},
                "passenger_info": {},
                "evidence": [{"field": "error", "quote": str(e)}],
            }

        data["_source_zip"] = src_zip
        data["_source_file"] = src_file
        extracted.append(data)
        
        if args.debug:
            debug_file = outdir / f"debug_{i}_output.json"
            debug_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    # Write JSONL
    jsonl_out = outdir / "docs_extracted.jsonl"
    with jsonl_out.open("w", encoding="utf-8") as f:
        for d in extracted:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"\n✅ Wrote: {jsonl_out}")

    # Flatten to CSV
    flat_rows = []
    for d in extracted:
        costs = d.get("costs", {})
        passenger = d.get("passenger_info", {})
        
        flat_rows.append({
            "source_zip": d.get("_source_zip"),
            "source_file": d.get("_source_file"),
            "doc_type": safe_get(d, "source_file_summary", "doc_type"),
            "category": safe_get(d, "source_file_summary", "category"),
            "vendor": safe_get(d, "source_file_summary", "vendor"),
            "country": safe_get(d, "source_file_summary", "country"),
            "city": safe_get(d, "source_file_summary", "city"),
            "trip_start_date": safe_get(d, "trip", "trip_start_date"),
            "trip_end_date": safe_get(d, "trip", "trip_end_date"),
            "origin": safe_get(d, "trip", "origin"),
            "destination": safe_get(d, "trip", "destination"),
            "passenger_name": passenger.get("passenger_name"),
            "ticket_number": passenger.get("ticket_number"),
            "currency": costs.get("currency"),
            "total_amount": costs.get("total_amount"),
            "air_travel_cost": costs.get("air_travel_cost"),
            "hotel_cost": costs.get("hotel_cost"),
            "local_transport_cost": costs.get("local_transport_cost"),
            "meal_cost": costs.get("meal_cost"),
            "tax_amount": costs.get("tax_amount"),
            "overall_description": d.get("overall_description"),
            "evidence_count": len(d.get("evidence") or []),
        })

    df_docs = pd.DataFrame(flat_rows)
    csv_docs_out = outdir / "docs_extracted.csv"
    df_docs.to_csv(csv_docs_out, index=False)
    print(f"✅ Wrote: {csv_docs_out}")

    # Trip aggregation
    agg_rows = []
    for d in extracted:
        k = trip_key(d)
        costs = d.get("costs") or {}
        
        # Calculate total from components
        total = to_float(costs.get("total_amount")) or sum(filter(None, [
            to_float(costs.get("air_travel_cost")),
            to_float(costs.get("hotel_cost")),
            to_float(costs.get("local_transport_cost")),
            to_float(costs.get("meal_cost"))
        ]))
        
        agg_rows.append({
            "trip_start_date": k[0],
            "trip_end_date": k[1],
            "destination": k[2],
            "source_file": d.get("_source_file"),
            "category": safe_get(d, "source_file_summary", "category"),
            "vendor": safe_get(d, "source_file_summary", "vendor"),
            "currency": costs.get("currency"),
            "total_amount": total,
            "air_travel_cost": to_float(costs.get("air_travel_cost")),
            "hotel_cost": to_float(costs.get("hotel_cost")),
            "local_transport_cost": to_float(costs.get("local_transport_cost")),
            "description": d.get("overall_description"),
        })

    df_agg = pd.DataFrame(agg_rows)

    # Group by trip
    df_agg["has_trip_info"] = df_agg[["trip_start_date", "trip_end_date", "destination"]].notna().any(axis=1)
    df_ok = df_agg[df_agg["has_trip_info"]].copy()

    if len(df_ok) > 0:
        grp = df_ok.groupby(["trip_start_date", "trip_end_date", "destination"], dropna=False)
        df_trips = grp.agg(
            currency=("currency", "first"),
            total_expenses=("total_amount", "sum"),
            airfare_total=("air_travel_cost", "sum"),
            hotel_total=("hotel_cost", "sum"),
            transport_total=("local_transport_cost", "sum"),
            doc_count=("source_file", "count"),
            vendors=("vendor", lambda x: ", ".join([str(v) for v in x if v])),
            source_files=("source_file", lambda x: "; ".join(sorted(set(x)))),
        ).reset_index()
    else:
        df_trips = pd.DataFrame(columns=[
            "trip_start_date","trip_end_date","destination","currency",
            "total_expenses","airfare_total","hotel_total","transport_total",
            "doc_count","vendors","source_files"
        ])

    trips_out = outdir / "trips_aggregated.csv"
    df_trips.to_csv(trips_out, index=False)
    print(f"✅ Wrote: {trips_out}")

    # Summary
    print("\n" + "="*60)
    print("EXTRACTION SUMMARY")
    print("="*60)
    print(f"Total documents processed: {len(docs)}")
    print(f"Documents with trip info: {df_agg['has_trip_info'].sum()}")
    print(f"Unique trips identified: {len(df_trips)}")
    print(f"\nDocument types:")
    print(df_docs['doc_type'].value_counts().to_string())
    print(f"\nCategories:")
    print(df_docs['category'].value_counts().to_string())
    
    if len(df_trips) > 0:
        print(f"\nTotal expenses: {df_trips['total_expenses'].sum():.2f} {df_trips['currency'].mode()[0] if len(df_trips['currency'].mode()) > 0 else 'N/A'}")

    print("\nDone! ✨")


if __name__ == "__main__":
    main()
