"""
output_writer.py
----------------
Serialises result rows to CSV and JSON.
"""
from __future__ import annotations

import csv
import json
import os
from typing import Any


def _csv_safe(value: Any) -> Any:
    """
    Ensure a value is a scalar or string safe for CSV.
    Lists and dicts are JSON-encoded to a single string.
    """
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def write_outputs(
    rows: list[dict[str, Any]],
    output_prefix: str,
) -> tuple[str, str]:
    """
    Write rows to <output_prefix>.csv and <output_prefix>.json.

    - CSV values that are lists/dicts are JSON-encoded strings.
    - JSON output keeps native Python types.
    - Parent directories are created automatically.
    - Returns (csv_path, json_path).
    """
    csv_path = output_prefix + ".csv"
    json_path = output_prefix + ".json"

    parent = os.path.dirname(csv_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    # Collect all fieldnames in sorted order (consistent with existing script).
    fieldnames = sorted({k for row in rows for k in row.keys()})

    # CSV: convert complex values to JSON strings.
    csv_rows = [{k: _csv_safe(v) for k, v in row.items()} for row in rows]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)

    # JSON: native types, pretty-printed.
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2, default=str)

    return csv_path, json_path
