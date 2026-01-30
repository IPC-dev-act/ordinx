#!/usr/bin/env python3
"""
Extract travel + expense info from OCR'd documents stored in a JSONL file.

Input JSONL format (one per document):
{
  "source_zip": "...",
  "source_file": "...",
  "text": "..."
}

Outputs:
- out/docs_extracted.jsonl
- out/docs_extracted.csv
- out/trips_aggregated.csv  (heuristic grouping by (trip_start, trip_end, destination))
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


# ----------------------------
# Robust JSON extraction
# ----------------------------
def extract_last_valid_json(text: str) -> Dict[str, Any]:
    """
    Extract the LAST valid JSON object from model output.
    This avoids picking up JSON templates echoed in the prompt.
    """
    candidates = re.findall(r"\{.*?\}", text, flags=re.DOTALL)
    parsed: List[Dict[str, Any]] = []
    for c in candidates:
        try:
            parsed.append(json.loads(c))
        except json.JSONDecodeError:
            continue
    if not parsed:
        raise ValueError("No valid JSON found in model output.\n---OUTPUT---\n" + text)
    return parsed[-1]


def clamp_text(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    # keep the beginning (usually contains header + key fields) + tail (often totals)
    head = s[: int(max_chars * 0.70)]
    tail = s[-int(max_chars * 0.30) :]
    return head + "\n...\n" + tail


# ----------------------------
# LLM prompt
# ----------------------------
def build_messages(doc_text: str) -> List[Dict[str, str]]:
    """
    Output schema matches user's needs:
    - trip_start_date, trip_end_date (ISO YYYY-MM-DD or null)
    - destination (city/country if possible)
    - air_travel_cost, hotel_cost, other_costs
    - short description
    - evidence quotes
    """
    system = (
        "You are a precise data extraction engine for travel/expense documents. "
        "Output ONLY valid JSON. No markdown. No explanations. "
        "Use null when unknown. Do not invent facts."
    )

    # NOTE: We include a JSON template in the prompt; parser takes LAST JSON from output.
    user = f"""Extract structured fields from the document text below.

Rules:
- Dates MUST be ISO format YYYY-MM-DD when present (e.g. 2024-02-04). If multiple, choose the trip start/end.
- If a flight itinerary shows outbound + return, trip_start_date = outbound date, trip_end_date = return date.
- If hotel stay is shown like "Séjour: 04/02/2024 - 07/02/2024", convert to ISO dates.
- Amounts: output numeric amounts as a string with dot decimal if you can (e.g. "280.20"). If only comma decimal exists, convert "14,00" -> "14.00".
- Currency: use 3-letter code if present (EUR, USD, etc.) else null.
- Classify category as one of: "air_travel", "hotel", "meal", "local_transport", "other".
- Document type: one of "flight_ticket", "hotel_invoice", "receipt", "other".

Return JSON with EXACTLY this structure:
{{
  "source_file_summary": {{
    "doc_type": null,
    "category": null,
    "vendor": null,
    "country": null,
    "city": null
  }},
  "trip": {{
    "trip_start_date": null,
    "trip_end_date": null,
    "origin": null,
    "destination": null
  }},
  "costs": {{
    "currency": null,
    "air_travel_cost": null,
    "hotel_cost": null,
    "other_costs": [
      {{"description": null, "amount": null, "currency": null}}
    ]
  }},
  "overall_description": null,
  "evidence": [
    {{"field": null, "quote": null}}
  ]
}}

Document text:
\"\"\"{doc_text}\"\"\"
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
def run_extract(tokenizer, model, doc_text: str, max_new_tokens: int = 400) -> Dict[str, Any]:
    messages = build_messages(doc_text)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
    )

    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    data = extract_last_valid_json(decoded)

    # Light normalization: ensure keys exist
    data.setdefault("source_file_summary", {})
    data.setdefault("trip", {})
    data.setdefault("costs", {})
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


def to_float_str(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if not s:
        return None
    # convert comma decimal to dot, remove spaces
    s = s.replace(" ", "").replace(",", ".")
    # keep digits + dot
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
    ap.add_argument("--model", default="models/qwen2.5-3b-instruct", help="Local model directory")
    ap.add_argument("--outdir", default="out", help="Output directory")
    ap.add_argument("--max-chars", type=int, default=8000, help="Max chars per doc passed to the model")
    ap.add_argument("--max-new-tokens", type=int, default=450, help="Max new tokens for generation")
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

        print(f"[{i}/{len(docs)}] Extracting: {src_file}")
        try:
            data = run_extract(tokenizer, model, text, max_new_tokens=args.max_new_tokens)
        except Exception as e:
            data = {
                "source_file_summary": {"doc_type": None, "category": None, "vendor": None, "country": None, "city": None},
                "trip": {"trip_start_date": None, "trip_end_date": None, "origin": None, "destination": None},
                "costs": {"currency": None, "air_travel_cost": None, "hotel_cost": None, "other_costs": []},
                "overall_description": None,
                "evidence": [{"field": "error", "quote": str(e)}],
            }

        data["_source_zip"] = src_zip
        data["_source_file"] = src_file
        extracted.append(data)

    # Write JSONL
    jsonl_out = outdir / "docs_extracted.jsonl"
    with jsonl_out.open("w", encoding="utf-8") as f:
        for d in extracted:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"✅ Wrote: {jsonl_out}")

    # Flatten to CSV-friendly
    flat_rows = []
    for d in extracted:
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
            "currency": safe_get(d, "costs", "currency"),
            "air_travel_cost": safe_get(d, "costs", "air_travel_cost"),
            "hotel_cost": safe_get(d, "costs", "hotel_cost"),
            "overall_description": d.get("overall_description"),
            "evidence_count": len(d.get("evidence") or []),
        })

    df_docs = pd.DataFrame(flat_rows)
    csv_docs_out = outdir / "docs_extracted.csv"
    df_docs.to_csv(csv_docs_out, index=False)
    print(f"✅ Wrote: {csv_docs_out}")

    # Trip aggregation (heuristic)
    agg_rows = []
    for d in extracted:
        k = trip_key(d)
        costs = d.get("costs") or {}
        agg_rows.append({
            "trip_start_date": k[0],
            "trip_end_date": k[1],
            "destination": k[2],
            "source_file": d.get("_source_file"),
            "currency": costs.get("currency"),
            "air_travel_cost": to_float_str(costs.get("air_travel_cost")),
            "hotel_cost": to_float_str(costs.get("hotel_cost")),
            "other_cost_sum": sum(
                [to_float_str(x.get("amount")) or 0.0 for x in (costs.get("other_costs") or [])]
            ),
            "description": d.get("overall_description"),
        })

    df_agg = pd.DataFrame(agg_rows)

    # group by trip key (ignore rows with all-null key)
    df_agg["trip_key_ok"] = df_agg[["trip_start_date", "trip_end_date", "destination"]].notna().any(axis=1)
    df_ok = df_agg[df_agg["trip_key_ok"]].copy()

    if len(df_ok) > 0:
        grp = df_ok.groupby(["trip_start_date", "trip_end_date", "destination"], dropna=False)
        df_trips = grp.agg(
            currency=("currency", "first"),
            airfare_total=("air_travel_cost", "sum"),
            hotel_total=("hotel_cost", "sum"),
            other_total=("other_cost_sum", "sum"),
            doc_count=("source_file", "count"),
            source_files=("source_file", lambda x: "; ".join(sorted(set(x)))),
            descriptions=("description", lambda x: " | ".join([s for s in x if isinstance(s, str) and s.strip()][:6])),
        ).reset_index()
    else:
        df_trips = pd.DataFrame(columns=[
            "trip_start_date","trip_end_date","destination","currency",
            "airfare_total","hotel_total","other_total","doc_count","source_files","descriptions"
        ])

    trips_out = outdir / "trips_aggregated.csv"
    df_trips.to_csv(trips_out, index=False)
    print(f"✅ Wrote: {trips_out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
