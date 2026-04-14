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

    Two formats are written for different downstream uses:
      CSV  — flat file suitable for spreadsheet analysis. Complex values
             (lists, dicts) are JSON-encoded into strings so every cell
             is a scalar. Columns are sorted alphabetically.
      JSON — pretty-printed, keeps native Python types (lists remain lists,
             dicts remain dicts). Better for programmatic analysis where you
             want to iterate over tool_call_log or url_checks as objects.

    Parent directories are created automatically.
    Returns (csv_path, json_path).
    """
    csv_path = output_prefix + ".csv"
    json_path = output_prefix + ".json"

    # Create the output directory if it does not already exist.
    parent = os.path.dirname(csv_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    # Collect the union of all keys across every row, then sort alphabetically.
    # This handles rows that may have different keys (e.g., an errored row that
    # was written before all output columns could be populated).
    fieldnames = sorted({k for row in rows for k in row.keys()})

    # CSV: convert lists/dicts to JSON strings so the file can be opened
    # in Excel or pandas without parsing errors.
    csv_rows = [{k: _csv_safe(v) for k, v in row.items()} for row in rows]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)

    # JSON: native types, pretty-printed. default=str handles any remaining
    # non-serialisable values (e.g. Python dataclasses) by converting to str.
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2, default=str)

    return csv_path, json_path
