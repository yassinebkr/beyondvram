"""CSV row builders and the per-block timeline writer for the OVR experiment."""
from __future__ import annotations

import csv
import math
from pathlib import Path

import bootstrap  # noqa: F401  (sys.path side effect)
from common import utc_now

BENCHMARK_ID = "OVR"
STAGES = ("read", "stage_copy", "h2d", "compute")


def _require_finite_nonnegative(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number")


def _require_positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _base(test: str, repetition: int, workload: str, seconds, value, unit: str, notes: str) -> dict:
    if not test or not workload or not unit:
        raise ValueError("test, workload, and unit must be non-empty")
    if repetition < 1:
        raise ValueError("repetition must be at least one")
    if seconds != "":
        _require_finite_nonnegative("seconds", float(seconds))
    _require_finite_nonnegative("value", float(value))
    return {
        "benchmark_id": BENCHMARK_ID,
        "test": test,
        "timestamp_utc": utc_now(),
        "repetition": repetition,
        "workload": workload,
        "seconds": seconds,
        "value": value,
        "unit": unit,
        "status": "ok",
        "notes": notes,
    }


def stage_row(variant: str, block_label: str, repetition: int, stage: str,
              seconds: float, blocks: int, block_bytes: int) -> dict:
    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage}")
    _require_positive("blocks", blocks)
    _require_positive("block_bytes", block_bytes)
    row = _base(f"{variant} stage {stage}", repetition, block_label, seconds, seconds, "s",
                f"sum over {blocks} blocks of {block_bytes} bytes; stage={stage}")
    row.update({"buffer_bytes": block_bytes, "chunk_bytes": block_bytes, "operations": blocks})
    return row


def makespan_row(variant: str, block_label: str, repetition: int,
                 seconds: float, blocks: int) -> dict:
    _require_positive("blocks", blocks)
    row = _base(f"{variant} makespan", repetition, block_label, seconds, seconds, "s",
                f"end-to-end wall time for {blocks} blocks")
    row["operations"] = blocks
    return row


def throughput_row(variant: str, block_label: str, repetition: int,
                   payload_bytes: int, seconds: float) -> dict:
    _require_positive("payload_bytes", payload_bytes)
    _require_positive("seconds", seconds)
    row = _base(f"{variant} payload throughput", repetition, block_label, seconds,
                payload_bytes / seconds / 1e9, "GB/s_payload",
                "payload bytes over end-to-end makespan")
    row["buffer_bytes"] = payload_bytes
    return row


def memory_row(variant: str, block_label: str, repetition: int, name: str,
               value_bytes: int, notes: str) -> dict:
    _require_finite_nonnegative("value_bytes", float(value_bytes))
    return _base(f"{variant} {name}", repetition, block_label, "", value_bytes, "bytes", notes)


def error_row(test: str, workload: str, reason: str) -> dict:
    return {"benchmark_id": BENCHMARK_ID, "test": test, "timestamp_utc": utc_now(),
            "workload": workload, "status": "error", "notes": reason}


def skipped_row(reason: str) -> dict:
    return {"benchmark_id": BENCHMARK_ID, "test": "overlap pipeline",
            "timestamp_utc": utc_now(), "status": "skipped", "notes": reason}


def write_timeline(path: Path, variant: str, block_label: str, repetition: int,
                   stage_times: dict[str, list[float]]) -> None:
    if repetition < 1:
        raise ValueError("repetition must be at least one")
    if tuple(stage_times) != STAGES:
        raise ValueError(f"timeline stages must be ordered as {STAGES}")
    lengths = {len(values) for values in stage_times.values()}
    if len(lengths) != 1:
        raise ValueError("every timeline stage must have one value per block")
    for stage, values in stage_times.items():
        for seconds in values:
            _require_finite_nonnegative(f"{stage} seconds", seconds)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(["variant", "block_label", "repetition", "block", "stage", "seconds"])
        for stage in STAGES:
            values = stage_times[stage]
            for block, seconds in enumerate(values):
                writer.writerow([variant, block_label, repetition, block, stage, f"{seconds:.9f}"])
