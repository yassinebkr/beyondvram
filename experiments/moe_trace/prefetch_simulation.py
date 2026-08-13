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

Output: results/moe-locality/prefetch-simulation.json
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
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


def main() -> None:
    traces = sorted(TRACE_DIR.glob("trace-*.jsonl"))
    totals = {r: [0, 0] for r in RESIDUAL_CAPACITIES}
    for path in traces:
        per_layer = load_trace(path)
        for r in RESIDUAL_CAPACITIES:
            rate, hits, total = simulate(per_layer, r)
            totals[r][0] += hits
            totals[r][1] += total
    result = {r: h / n for r, (h, n) in totals.items()}
    print("combined predictor + LRU-residual hit rates (pooled, all layers):")
    for r, rate in result.items():
        print(f"  residual {r:3d} (+8 prefetched): {rate:.4f}")
    OUT.write_text(json.dumps({
        "scheme": "predict E(t+1)=E(t), LRU residual per layer",
        "residual_capacities": RESIDUAL_CAPACITIES,
        "hit_rates": {str(r): v for r, v in result.items()},
        "traces": [p.name for p in traces],
    }, indent=1), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
