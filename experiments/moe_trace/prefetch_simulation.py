"""Prefetch-prediction simulation: hit rate of 'predict next token = previous token's experts'.

Design input for the async-prefetch cache (v2). The v1 sync cache failed on
resolve overhead and cap-8 thrash; v2 prefetches the PREDICTED set (previous
token's 8 experts per layer, measured adjacent overlap ~0.45) during idle
windows and keeps an LRU residual cache for the rest. This script measures,
from the real traces:

  - prediction accuracy: |E(t) ∩ E(t-1)| / 8 per layer (already known ~0.45);
  - end-to-end hit rate of the combined scheme: request hits if the expert is
    either in the predicted set (prefetched) or in the LRU residual of
    capacity R per layer;
  - for several residual capacities R.

Progress banners print per trace file. Ctrl+C writes partial results with
status="interrupted" and exits with code 130.

Output: results/moe-locality/prefetch-simulation.json
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "system"))
from common import ts  # noqa: E402

TRACE_DIR = ROOT / "results/moe-locality"
OUT = TRACE_DIR / "prefetch-simulation.json"
RESIDUAL_CAPACITIES = [4, 8, 16, 24]


def load_trace(path: Path) -> dict[int, list[tuple[int, tuple[int, ...]]]]:
    per_layer: dict[int, list[tuple[int, tuple[int, ...]]]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                r = json.loads(line)
                per_layer[r["layer"]].append((r["pos"], tuple(r["experts"])))
    for records in per_layer.values():
        records.sort()
    return dict(per_layer)


def simulate(per_layer, residual_cap: int) -> tuple[float, int, int]:
    """Combined predictor+residual hit rate over all layers. Returns (rate, hits, total)."""
    hits = total = 0
    for records in per_layer.values():
        predicted: set[int] = set()
        residual: list[int] = []  # LRU, MRU first
        for _, experts in records:
            current = set(experts)
            for expert in current:
                total += 1
                if expert in predicted or expert in residual:
                    hits += 1
            # residual LRU update with actual requests
            for expert in experts:
                if expert in residual:
                    residual.remove(expert)
                residual.insert(0, expert)
            del residual[residual_cap:]
            predicted = current  # next token's prediction = this token's set
    return (hits / total if total else 0.0), hits, total


def write_payload(traces: list[Path], totals: dict[int, list[int]],
                  status: str) -> dict:
    result = {r: (h / n if n else None) for r, (h, n) in totals.items()}
    payload = {
        "status": status,
        "scheme": "predict E(t+1)=E(t), LRU residual per layer",
        "residual_capacities": RESIDUAL_CAPACITIES,
        "hit_rates": {str(r): v for r, v in result.items()},
        "traces": [p.name for p in traces],
    }
    OUT.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return payload


def main() -> None:
    traces = sorted(TRACE_DIR.glob("trace-*.jsonl"))
    if not traces:
        raise SystemExit(f"no trace files found in {TRACE_DIR}")
    print(f"[{ts()}] prefetch simulation: {len(traces)} trace file(s), "
          f"residual capacities {RESIDUAL_CAPACITIES}", flush=True)
    print(f"[{ts()}] output -> {OUT} (Ctrl+C writes partial results)",
          flush=True)

    totals = {r: [0, 0] for r in RESIDUAL_CAPACITIES}
    processed: list[Path] = []
    try:
        for i, path in enumerate(traces, 1):
            print(f"[{ts()}] [file {i}/{len(traces)}] {path.stem}: simulating",
                  flush=True)
            per_layer = load_trace(path)
            for r in RESIDUAL_CAPACITIES:
                rate, hits, total = simulate(per_layer, r)
                totals[r][0] += hits
                totals[r][1] += total
            processed.append(path)
            print(f"[{ts()}] [file {i}/{len(traces)}] {path.stem}: {total} "
                  f"requests x {len(RESIDUAL_CAPACITIES)} capacities",
                  flush=True)
    except KeyboardInterrupt:
        write_payload(processed, totals, status="interrupted")
        print(f"\n[{ts()}] interrupted — partial results ({len(processed)}/"
              f"{len(traces)} files) -> {OUT}", flush=True)
        sys.exit(130)

    payload = write_payload(traces, totals, status="complete")
    print("combined predictor + LRU-residual hit rates (pooled, all layers):",
          flush=True)
    for r, rate in payload["hit_rates"].items():
        print(f"  residual {int(r):3d} (+8 prefetched): {rate:.4f}", flush=True)
    print(f"[{ts()}] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
