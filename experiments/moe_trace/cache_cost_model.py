"""Cache cost model: would a VRAM expert cache beat CPU expert compute?

Joins three measured inputs:
- per-layer LRU hit rates from results/moe-locality/locality-analysis.json
  (traced Qwen3-30B-A3B routing, 10 prompts x 128 tokens);
- exact per-expert byte sizes from the Q4_K_M GGUF;
- measured system numbers: pinned/pageable H2D bandwidth (B05/B04) and the
  llama.cpp b10355 generation anchors (25.15 tok/s at -ngl 18; 16.87 tok/s
  with --n-cpu-moe 48 at the same layer split).

Timing decomposition (ms/token), all measured or derived from measured points:
  T(ngl18)            = A + 30c + 18g          = 39.76   (25.15 tok/s)
  T(ngl18, cpu-moe48) = A + 48c                = 59.28   (16.87 tok/s)
  => c - g = 1.084 ms per layer per token
where c = CPU expert-compute time for one MoE layer, g = GPU expert-compute
time for one MoE layer, A = attention + overhead for the -ngl 18 split.

For a hypothetical expert cache on layers 18..47 (the 30 layers whose experts
are CPU-computed today), the saving per cached layer per token is
  saving_l(C) = (c - g) - transfer_l(C) = 1.084 - misses_l(C) * expert_bytes_l / BW
assuming synchronous (non-overlapped) transfers — the conservative case. An
ideal-overlap column assumes transfer hides fully under CPU attention compute.

Nothing here is measured end-to-end; it is arithmetic over measured inputs,
with a sensitivity range on bandwidth (pageable vs pinned) and on the
(c - g) anchor (+/-20%).

Progress banners print per step. Ctrl+C writes the rows computed so far with
status="interrupted" and exits with code 130.

Output: results/moe-locality/cache-cost-model.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from gguf import GGUFReader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "system"))
from common import ts  # noqa: E402
MODEL = ROOT / "models" / "Qwen3-30B-A3B-GGUF" / "Qwen3-30B-A3B-Q4_K_M.gguf"
ANALYSIS = ROOT / "results" / "moe-locality" / "locality-analysis.json"
OUT = ROOT / "results" / "moe-locality" / "cache-cost-model.json"

CACHED_LAYERS = list(range(0, 30))  # experts CPU-computed at the -ngl 18 optimum
                                    # (-ngl 18 offloads the LAST 18 layers, 31-47)
C_G_ANCHOR_MS = (1000 / 16.87 - 1000 / 25.15) / 18  # c - g, measured
T_BASELINE_MS = 1000 / 25.15
PINNED_GBS = 26.06   # B05 median
PAGEABLE_GBS = 11.10  # B04 median


def expert_bytes_per_layer() -> dict[int, float]:
    """Per-layer bytes of one expert (gate+up+down exps tensors / 128)."""
    per_layer: dict[int, int] = defaultdict(int)
    reader = GGUFReader(str(MODEL))
    for tensor in reader.tensors:
        if ".ffn_" in tensor.name and "_exps" in tensor.name:
            per_layer[int(tensor.name.split(".")[1])] += tensor.n_bytes
    return {layer: total / 128 for layer, total in per_layer.items()}


def mean_hit_rates(analysis: dict, capacity: int) -> dict[int, float]:
    """Request-weighted mean hit rate per layer across all trace files."""
    hits: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for per_layer in analysis["lru_hit_rates"]:
        for layer, caps in per_layer.items():
            entry = caps[str(capacity)]
            if entry["hit_rate"] is not None:
                hits[int(layer)][0] += entry["hit_rate"] * entry["requests"]
                hits[int(layer)][1] += entry["requests"]
    return {layer: h / n for layer, (h, n) in hits.items() if n}


def write_payload(rows: list[dict], exp_bytes: dict[int, float],
                  budgets_gb: list[float], status: str) -> None:
    payload = {
        "status": status,
        "inputs": {
            "t_baseline_ms_ngl18": T_BASELINE_MS,
            "c_minus_g_ms_per_layer": C_G_ANCHOR_MS,
            "pinned_h2d_gb_s": PINNED_GBS,
            "pageable_h2d_gb_s": PAGEABLE_GBS,
            "cached_layers": CACHED_LAYERS,
            "expert_mb_per_layer": {str(l): round(exp_bytes[l] / 1e6, 3)
                                    for l in CACHED_LAYERS if l in exp_bytes},
            "cache_vram_budgets_gb": budgets_gb,
        },
        "rows": rows,
        "notes": [
            "est_tok_s_sync assumes synchronous miss transfers (no overlap).",
            "ideal_overlap column is the compute-only ceiling (transfer fully hidden).",
            "cache_footprint_gb is per-layer capacity x expert bytes x 30 layers; "
            "it must fit in free VRAM alongside the -ngl 18 placement.",
        ],
    }
    OUT.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-vram-gb", type=float, nargs="+",
                        default=[0.5, 1.0, 2.0, 4.0],
                        help="VRAM budgets for the expert cache")
    args = parser.parse_args()

    print(f"[{ts()}] cache cost model: {ANALYSIS.name} + per-expert bytes "
          f"from {MODEL.name}", flush=True)
    print(f"[{ts()}] output -> {OUT} (Ctrl+C writes partial results)",
          flush=True)

    exp_bytes: dict[int, float] = {}
    rows: list[dict] = []
    try:
        print(f"[{ts()}] [step 1/2] loading hit rates and GGUF expert bytes",
              flush=True)
        analysis = json.loads(ANALYSIS.read_text(encoding="utf-8"))
        exp_bytes = expert_bytes_per_layer()
        capacities = [int(c) for c in analysis["lru_capacities"]]

        print(f"[{ts()}] [step 2/2] scoring {len(capacities)} capacities x "
              f"2 bandwidths x 3 anchor scales", flush=True)
        for capacity in capacities:
            hit = mean_hit_rates(analysis, capacity)
            # misses and bytes summed over the 30 cached layers, per token
            misses = {l: 8 * (1.0 - hit.get(l, 0.0)) for l in CACHED_LAYERS}
            bytes_per_token = sum(misses[l] * exp_bytes[l] for l in CACHED_LAYERS)
            cache_footprint_gb = sum(capacity * exp_bytes[l] for l in CACHED_LAYERS) / 1e9
            for bw_name, bw in (("pinned", PINNED_GBS), ("pageable", PAGEABLE_GBS)):
                transfer_ms = bytes_per_token / (bw * 1e9) * 1000
                for anchor_scale in (0.8, 1.0, 1.2):
                    anchor = C_G_ANCHOR_MS * anchor_scale
                    saving = sum(max(0.0, anchor - misses[l] * exp_bytes[l] / (bw * 1e9) * 1000)
                                 for l in CACHED_LAYERS)
                    t_sync = T_BASELINE_MS - saving
                    rows.append({
                        "capacity": capacity,
                        "bandwidth": bw_name,
                        "anchor_scale": anchor_scale,
                        "mean_misses_per_token": sum(misses.values()) / len(CACHED_LAYERS),
                        "bytes_per_token_mb": bytes_per_token / 1e6,
                        "transfer_ms_per_token": transfer_ms,
                        "cache_footprint_gb": cache_footprint_gb,
                        "est_ms_per_token_sync": t_sync,
                        "est_tok_s_sync": 1000 / t_sync if t_sync > 0 else None,
                        "est_tok_s_ideal_overlap": 1000 / max(T_BASELINE_MS - sum(
                            max(0.0, anchor) for l in CACHED_LAYERS), 1.0),
                    })
    except KeyboardInterrupt:
        write_payload(rows, exp_bytes, args.cache_vram_gb, status="interrupted")
        print(f"\n[{ts()}] interrupted — partial results ({len(rows)} rows) "
              f"-> {OUT}", flush=True)
        sys.exit(130)

    write_payload(rows, exp_bytes, args.cache_vram_gb, status="complete")

    print(f"c - g anchor: {C_G_ANCHOR_MS:.3f} ms/layer/token (measured)",
          flush=True)
    print(f"expert size: {exp_bytes[18]/1e6:.2f} MB (layer 18); "
          f"baseline {1000/T_BASELINE_MS:.2f} tok/s\n", flush=True)
    print(f"{'cap':>4} {'miss/tok':>8} {'MB/tok':>7} {'xfer ms':>8} {'GB vram':>8} "
          f"{'tok/s sync(pinned)':>18} {'tok/s ideal':>11}", flush=True)
    for row in rows:
        if row["bandwidth"] == "pinned" and row["anchor_scale"] == 1.0:
            print(f"{row['capacity']:>4} {row['mean_misses_per_token']:>8.2f} "
                  f"{row['bytes_per_token_mb']:>7.1f} {row['transfer_ms_per_token']:>8.2f} "
                  f"{row['cache_footprint_gb']:>8.2f} {row['est_tok_s_sync']:>18.1f} "
                  f"{row['est_tok_s_ideal_overlap']:>11.1f}", flush=True)
    print(f"\n[{ts()}] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
