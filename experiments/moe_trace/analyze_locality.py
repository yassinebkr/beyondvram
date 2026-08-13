"""Analyze MoE expert-routing locality from moe-trace JSONL files.

Inputs are JSONL files with one record per (layer, token column):
    {"pos": int, "layer": int, "experts": [int, ...]}

Reports, per trace file and pooled across files:
    - per-layer expert activation frequency (counts, top-share, entropy);
    - consecutive-token expert-set overlap (mean |E(t) & E(t+1)| per layer);
    - per-layer expert reuse-distance distribution in tokens (percentiles);
    - hypothetical per-layer LRU cache hit rates for several capacities.

Raw aggregates (counts, histograms) are written alongside derived metrics;
nothing is extrapolated beyond the recorded traces.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRACE_DIR = ROOT / "results" / "moe-locality"

OVERLAP_DISTANCES = [1, 2, 4, 8, 16]
LRU_CAPACITIES = [8, 12, 16, 24, 32, 48, 64, 96, 128]
REUSE_HIST_MAX = 512  # distances >= this bucket into the last bin


def load_trace(path: Path) -> dict[int, list[tuple[int, tuple[int, ...]]]]:
    """Return layer -> [(pos, experts tuple), ...] sorted by position."""
    per_layer: dict[int, list[tuple[int, tuple[int, ...]]]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            per_layer[record["layer"]].append((record["pos"], tuple(record["experts"])))
    for records in per_layer.values():
        records.sort(key=lambda item: item[0])
    return dict(per_layer)


def expert_frequency(per_layer) -> dict:
    """Per-layer activation counts, top-1/top-8 share, and entropy."""
    result = {}
    for layer, records in sorted(per_layer.items()):
        counts: dict[int, int] = defaultdict(int)
        total = 0
        for _, experts in records:
            for expert in set(experts):  # count each expert once per token
                counts[expert] += 1
                total += 1
        ranked = sorted(counts.values(), reverse=True)
        entropy = -sum((c / total) * math.log2(c / total) for c in counts.values() if c)
        result[layer] = {
            "tokens": len(records),
            "distinct_experts_used": len(counts),
            "top1_share": ranked[0] / total if total else 0.0,
            "top8_share": sum(ranked[:8]) / total if total else 0.0,
            "entropy_bits": entropy,
            "counts": {str(k): v for k, v in sorted(counts.items())},
        }
    return result


def consecutive_overlap(per_layer) -> dict:
    """Mean expert-set overlap between tokens at exact position distances.

    Position-aware: pairs are formed only when both positions exist, which
    matters for the last MoE layer (prompt-region positions without requested
    logits are never routed by llama.cpp and are absent from the trace).
    """
    result = {}
    for layer, records in sorted(per_layer.items()):
        sets_by_pos = {pos: set(experts) for pos, experts in records}
        layer_result = {}
        for distance in OVERLAP_DISTANCES:
            overlaps = [
                len(sets_by_pos[pos] & sets_by_pos[pos + distance]) / len(sets_by_pos[pos])
                for pos in sets_by_pos
                if pos + distance in sets_by_pos
            ]
            layer_result[distance] = {
                "mean_overlap": sum(overlaps) / len(overlaps) if overlaps else None,
                "pairs": len(overlaps),
            }
        result[layer] = layer_result
    return result


def reuse_distance(per_layer) -> dict:
    """Per-layer reuse distance in token positions since previous activation."""
    result = {}
    for layer, records in sorted(per_layer.items()):
        last_seen: dict[int, int] = {}
        histogram = [0] * (REUSE_HIST_MAX + 1)
        distances = []
        for pos, experts in records:
            for expert in set(experts):
                if expert in last_seen:
                    distance = pos - last_seen[expert]
                    distances.append(distance)
                    histogram[min(distance, REUSE_HIST_MAX)] += 1
                last_seen[expert] = pos
        ordered = sorted(distances)
        result[layer] = {
            "reuse_events": len(distances),
            "median": ordered[len(ordered) // 2] if ordered else None,
            "p90": ordered[int(0.9 * (len(ordered) - 1))] if ordered else None,
            "histogram": histogram,
        }
    return result


def lru_hit_rates(per_layer) -> dict:
    """Simulated per-layer LRU expert-cache hit rates for several capacities."""
    result = {}
    for layer, records in sorted(per_layer.items()):
        requests = [list(dict.fromkeys(experts)) for _, experts in records]
        layer_result = {}
        for capacity in LRU_CAPACITIES:
            cache: list[int] = []  # most-recently-used first
            hits = 0
            total = 0
            for experts in requests:
                for expert in experts:
                    total += 1
                    if expert in cache:
                        hits += 1
                        cache.remove(expert)
                    cache.insert(0, expert)
                del cache[capacity:]
            layer_result[capacity] = {
                "hit_rate": hits / total if total else None,
                "requests": total,
            }
        result[layer] = layer_result
    return result


def pooled_lru(per_layer_results: list[dict]) -> dict:
    """Request-weighted mean hit rate across layers and trace files."""
    totals: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for per_layer in per_layer_results:
        for layer_results in per_layer.values():
            for capacity, entry in layer_results.items():
                totals[capacity][0] += round(entry["hit_rate"] * entry["requests"])
                totals[capacity][1] += entry["requests"]
    return {capacity: hits / total if total else None
            for capacity, (hits, total) in sorted(totals.items())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, nargs="+", default=None,
                        help="trace JSONL files (default: all trace-*.jsonl in results/moe-locality)")
    args = parser.parse_args()

    traces = args.traces or sorted(TRACE_DIR.glob("trace-*.jsonl"))
    if not traces:
        raise SystemExit(f"no trace files found in {TRACE_DIR}")

    all_frequency, all_overlap, all_reuse, all_lru = [], [], [], []
    per_file_summary = {}
    for trace_path in traces:
        per_layer = load_trace(trace_path)
        name = trace_path.stem
        frequency = expert_frequency(per_layer)
        overlap = consecutive_overlap(per_layer)
        reuse = reuse_distance(per_layer)
        lru = lru_hit_rates(per_layer)
        all_frequency.append(frequency)
        all_overlap.append(overlap)
        all_reuse.append(reuse)
        all_lru.append(lru)

        tokens = sum(len(v) for v in per_layer.values()) // max(len(per_layer), 1)
        pooled_overlap1 = [
            entry[1]["mean_overlap"] * entry[1]["pairs"]
            for entry in overlap.values()
            if entry[1]["mean_overlap"] is not None
        ]
        overlap1_pairs = sum(entry[1]["pairs"] for entry in overlap.values())
        per_file_summary[name] = {
            "tokens_per_layer": tokens,
            "layers": len(per_layer),
            "mean_consecutive_overlap": (sum(pooled_overlap1) / overlap1_pairs
                                         if overlap1_pairs else None),
        }
        print(f"{name}: {len(per_layer)} layers, {tokens} tokens/layer, "
              f"mean consecutive overlap {per_file_summary[name]['mean_consecutive_overlap']:.3f}")

    pooled = pooled_lru(all_lru)
    print("\npooled per-layer LRU hit rates (request-weighted):")
    for capacity, hit_rate in pooled.items():
        print(f"  capacity {capacity:4d}: {hit_rate:.4f}")

    out_path = TRACE_DIR / "locality-analysis.json"
    payload = {
        "traces": [p.name for p in traces],
        "overlap_distances": OVERLAP_DISTANCES,
        "lru_capacities": LRU_CAPACITIES,
        "per_file_summary": per_file_summary,
        "pooled_lru_hit_rates": {str(k): v for k, v in pooled.items()},
        "expert_frequency": all_frequency,
        "consecutive_overlap": all_overlap,
        "reuse_distance": all_reuse,
        "lru_hit_rates": all_lru,
    }
    out_path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
