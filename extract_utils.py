"""Shared utilities for extraction scripts."""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


def normalize_date_to_iso(date_str: str) -> Optional[str]:
    """Convert various date formats to ISO YYYY-MM-DD."""
    if not date_str or not isinstance(date_str, str):
        return None

    date_str = date_str.strip()

    if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
        return date_str

    match = re.match(r"^(\d{1,2})[/\.](\d{1,2})[/\.](\d{4})$", date_str)
    if match:
        day, month, year = match.groups()
        try:
            dt = datetime(int(year), int(month), int(day))
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass

    month_names = {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
        "janvier": 1,
        "février": 2,
        "mars": 3,
        "avril": 4,
        "mai": 5,
        "juin": 6,
        "juillet": 7,
        "août": 8,
        "septembre": 9,
        "octobre": 10,
        "novembre": 11,
        "décembre": 12,
    }

    match = re.match(r"^(\d{1,2})\s+([a-zé]+)\s+(\d{4})$", date_str.lower())
    if match:
        day, month_str, year = match.groups()
        month = month_names.get(month_str[:3])
        if month:
            try:
                dt = datetime(int(year), month, int(day))
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                pass

    return None


def extract_last_valid_json(text: str) -> Dict[str, Any]:
    """Extract the LAST valid JSON object from model output."""
    candidates = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, flags=re.DOTALL)

    parsed: List[Dict[str, Any]] = []
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                parsed.append(obj)
        except json.JSONDecodeError:
            continue

    if not parsed:
        md_json = re.findall(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
        for c in md_json:
            try:
                parsed.append(json.loads(c))
            except json.JSONDecodeError:
                continue

    if not parsed:
        raise ValueError("No valid JSON found in model output.\n---OUTPUT---\n" + text[:1000])

    return parsed[-1]


def clamp_text(s: str, max_chars: int, head_ratio: float = 0.85) -> str:
    """Smart text truncation that preserves key information."""
    s = (s or "").strip()
    if len(s) <= max_chars or len(s) < 200:
        return s

    head = s[: int(max_chars * head_ratio)]
    tail = s[-int(max_chars * (1 - head_ratio)) :]
    return head + "\n[... truncated ...]\n" + tail


def trip_key(d: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    t = d.get("trip") or {}
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
    s = s.replace(" ", "").replace(",", ".")
    s = re.sub(r"[€$£¥₹]", "", s)
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
