"""Shared result-writing and live-subprocess helpers for characterization runs."""

from __future__ import annotations

import csv
import json
import math
import statistics
import subprocess
import threading
import time
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


def ts() -> str:
    """Local HH:MM:SS timestamp for progress banners."""
    return time.strftime("%H:%M:%S")


def terminate_child(proc: subprocess.Popen) -> None:
    """Terminate a child process, escalating to kill after a grace period."""
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def run_live(cmd: list[str], label: str, quiet: bool = False,
             timeout: int = 3600) -> dict:
    """Run cmd, streaming child stdout/stderr live with a `  | ` prefix.

    Returns {"rc", "stdout", "stderr", "wall_s"}. Ctrl+C terminates the child
    and re-raises so the caller can persist partial results. A timeout also
    terminates the child and reports rc=-1.
    """
    print(f"[{ts()}] {label} launch: {Path(str(cmd[0])).name} "
          f"{' '.join(str(a) for a in cmd[1:3])} ...", flush=True)
    proc = subprocess.Popen(
        [str(c) for c in cmd], stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, errors="replace")
    parts: dict[str, list[str]] = {"out": [], "err": []}

    def drain(stream, key: str) -> None:
        for line in stream:
            parts[key].append(line)
            if not quiet:
                print(f"  | {line}", end="", flush=True)

    threads = [
        threading.Thread(target=drain, args=(proc.stdout, "out"), daemon=True),
        threading.Thread(target=drain, args=(proc.stderr, "err"), daemon=True),
    ]
    start = time.perf_counter()
    try:
        for t in threads:
            t.start()
        rc = proc.wait(timeout=timeout)
    except KeyboardInterrupt:
        print(f"\n[{ts()}] {label} interrupt — terminating child", flush=True)
        terminate_child(proc)
        raise
    except subprocess.TimeoutExpired:
        print(f"\n[{ts()}] {label} timeout ({timeout}s) — terminating child",
              flush=True)
        terminate_child(proc)
        rc = -1
    for t in threads:
        t.join(timeout=5)
    wall = round(time.perf_counter() - start, 1)
    print(f"[{ts()}] {label} done rc={rc} wall={wall}s", flush=True)
    return {"rc": rc, "stdout": "".join(parts["out"]),
            "stderr": "".join(parts["err"]), "wall_s": wall}

