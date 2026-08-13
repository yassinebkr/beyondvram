"""Shared result-writing helpers for system characterization benchmarks."""

from __future__ import annotations

import csv
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "system"
RAW_CSV = RESULTS_DIR / "raw_measurements.csv"
SUMMARY_CSV = ROOT / "results" / "system_characterization.csv"

FIELDS = [
    "benchmark_id",
    "test",
    "timestamp_utc",
    "repetition",
    "workload",
    "buffer_bytes",
    "chunk_bytes",
    "operations",
    "seconds",
    "value",
    "unit",
    "status",
    "notes",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_rows(path: Path, rows: Iterable[dict], append: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    mode = "a" if append else "w"
    with path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        if not exists or not append:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def skipped_row(benchmark_id: str, test: str, reason: str) -> dict:
    return {
        "benchmark_id": benchmark_id,
        "test": test,
        "timestamp_utc": utc_now(),
        "status": "skipped",
        "notes": reason,
    }


def rebuild_summary() -> None:
    """Build one median row per benchmark/workload while retaining skip records."""
    if not RAW_CSV.exists():
        return
    with RAW_CSV.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    groups: dict[tuple[str, str, str, str], list[dict]] = {}
    skipped: dict[tuple[str, str, str, str], dict] = {}
    for row in rows:
        key = (row["benchmark_id"], row["test"], row["workload"], row["unit"])
        if row["status"] == "ok" and row["value"]:
            groups.setdefault(key, []).append(row)
        elif key not in groups:
            skipped[key] = row

    summary = []
    for key, samples in sorted(groups.items()):
        values = [float(sample["value"]) for sample in samples]
        seconds = [float(sample["seconds"]) for sample in samples if sample["seconds"]]
        first = samples[0]
        summary.append(
            {
                **first,
                "timestamp_utc": utc_now(),
                "repetition": "median",
                "seconds": statistics.median(seconds) if seconds else "",
                "value": statistics.median(values),
                "status": "ok",
                "notes": (
                    f"n={len(values)}; min={min(values):.6g}; max={max(values):.6g}; "
                    f"stdev={statistics.stdev(values):.6g}" if len(values) > 1 else "n=1"
                ),
            }
        )
    measured_keys = set(groups)
    measured_ids = {key[0] for key in groups}
    for key, row in sorted(skipped.items()):
        if key not in measured_keys and row["benchmark_id"] not in measured_ids:
            summary.append(row)
    write_rows(SUMMARY_CSV, summary, append=False)


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

