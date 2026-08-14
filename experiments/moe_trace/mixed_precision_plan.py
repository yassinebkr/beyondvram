"""Next-experiments #4: per-layer mixed-precision expert plan (bytes + roofline).

Constraint discovered 2026-08-14: GGUF packs all 128 experts of a layer into
one tensor per matrix kind (blk.N.ffn_{gate,up,down}_exps.weight), so mixed
precision is expressible per LAYER, not per expert. Reads the local Q4_K_M
GGUF for measured per-layer expert bytes, computes IQ2_XXS replacement sizes
from tensor shapes and ggml block sizes, and evaluates split scenarios
(keep the first/last K layers at Q4_K_M — the quality-sensitivity prior —
quantize the middle to IQ2_XXS). Routing entropy per layer from the Track-1
locality analysis is recorded as a hypothesis marker only; the perplexity
gate (experiments/low_bit/perplexity_quants.py) arbitrates quality.

Placement math: dense/shared tensors go to VRAM first (measured Track-1
optimum had 15 expert layers VRAM-resident = ~5.9 GB), then expert layers
fill the remaining budget; every expert layer reads top-8 experts per token
regardless of routing, so per-layer bytes/token is uniform within a quant.

Progress banners print per step. Ctrl+C writes the sections computed so far
with status="interrupted" and exits with code 130.

Output: results/moe-locality/mixed-precision-plan.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from gguf import GGUFReader
from gguf.constants import GGMLQuantizationType

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "system"))
from common import ts  # noqa: E402
MODEL = ROOT / "models/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf"
LOCALITY = ROOT / "results/moe-locality/locality-analysis.json"
OUT = ROOT / "results/moe-locality/mixed-precision-plan.json"

TOP_K = 8
N_LAYERS = 48
N_EXPERTS = 128
# Effective DDR4 read throughput implied by the all-experts-CPU gpt-oss
# measurement, confirmed as this machine's constant (docs/next-experiments.md
# experiment 1). Measured tg at the Track-1 optimum ran ~0.83x the same
# roofline, so predictions apply that calibration.
DDR4_READ_GBS = 26.6
ROOFLINE_CALIBRATION = 0.83
# VRAM budget for expert layers at the Track-1 optimum (-ngl 48
# --n-cpu-moe 33): 15 Q4_K_M expert layers = 15 x 391.6 MB.
VRAM_EXPERT_BUDGET_GB = 15 * 391.6 / 1000

# ggml block sizes (bytes per block, weights per block) for the quants involved.
BLOCK = {  # type: (block_bytes, block_weights)
    "Q4_K": (144, 256),
    "Q6_K": (210, 256),
    "IQ2_XXS": (66, 256),  # 2.0625 bpw
}


def bytes_at(tensor_shape: list[int], type_name: str) -> int:
    n_weights = 1
    for d in tensor_shape:
        n_weights *= int(d)  # int(): gguf shapes are numpy uint; keep Python ints
    block_bytes, block_weights = BLOCK[type_name]
    assert n_weights % block_weights == 0
    return n_weights // block_weights * block_bytes


def build_payload(q4: dict[int, int], iq2: dict[int, int],
                  entropy_mean: dict[int, float], scenarios: list[dict],
                  baseline: dict | None, status: str) -> dict:
    """Assemble the output payload; sections not yet computed stay null/empty."""
    measured = None
    if q4 and iq2:
        measured = {
            "q4km_expert_mb_q6k_down": round(max(q4.values()) / 1e6, 1),
            "q4km_expert_mb_q4k_down": round(min(q4.values()) / 1e6, 1),
            "iq2xxs_expert_mb_gate_up": round(iq2[0] / 1e6 / 3, 1),
            "note": "official Q4_K_M alternates the down-proj quant by layer; "
                    "IQ2_XXS sizes computed from shapes at 2.0625 bpw",
        }
    return {
        "status": status,
        "model": MODEL.name,
        "measured_per_layer": measured,
        "constants": {
            "top_k": TOP_K, "n_experts": N_EXPERTS, "n_layers": N_LAYERS,
            "ddr4_read_gbs": DDR4_READ_GBS, "roofline_calibration": ROOFLINE_CALIBRATION,
            "vram_expert_budget_gb": round(VRAM_EXPERT_BUDGET_GB, 2),
        },
        "baseline_q4km": baseline,
        "scenarios_middle_iq2": scenarios,
        "routing_entropy_bits_per_layer": {str(k): round(v, 3) for k, v in entropy_mean.items()},
        "entropy_note": "max = 7.0 bits (128 experts); hypothesis marker for quality "
                        "sensitivity only — the perplexity gate arbitrates",
    }


def main() -> None:
    print(f"[{ts()}] mixed-precision plan: {MODEL.name}, {N_LAYERS} layers, "
          f"keep-per-end scenarios 12/8/4/0", flush=True)
    print(f"[{ts()}] output -> {OUT} (Ctrl+C writes partial results)",
          flush=True)

    q4: dict[int, int] = {}
    iq2: dict[int, int] = {}
    entropy_mean: dict[int, float] = {}
    scenarios: list[dict] = []
    baseline: dict | None = None
    try:
        print(f"[{ts()}] [step 1/3] reading per-layer expert bytes from the "
              f"GGUF", flush=True)
        reader = GGUFReader(str(MODEL))
        per_layer: dict[int, dict[str, int]] = {}
        for t in reader.tensors:
            if "exps" not in t.name:
                continue
            layer = int(t.name.split(".")[1])
            shape = list(t.shape)
            kind = t.name.split(".")[2]  # ffn_gate_exps / ffn_up_exps / ffn_down_exps
            entry = per_layer.setdefault(layer, {"q4km_bytes": 0, "iq2_bytes": 0})
            entry["q4km_bytes"] += t.n_bytes
            entry["iq2_bytes"] += bytes_at(shape, "IQ2_XXS")

        assert len(per_layer) == N_LAYERS, f"expected {N_LAYERS} layers, got {len(per_layer)}"
        # The official Q4_K_M mix is not uniform: 24 layers carry Q6_K down-proj
        # (391.6 MB) and 24 carry Q4_K down-proj (339.7 MB). Per-layer actuals.
        q4 = {k: v["q4km_bytes"] for k, v in per_layer.items()}
        iq2 = {k: v["iq2_bytes"] for k, v in per_layer.items()}

        print(f"[{ts()}] [step 2/3] loading routing-entropy hypothesis markers",
              flush=True)
        # Routing entropy per layer (hypothesis marker only).
        loc = json.loads(LOCALITY.read_text(encoding="utf-8"))
        entropy = {}
        for per_file in loc["expert_frequency"]:
            for layer_str, stats in per_file.items():
                entropy.setdefault(int(layer_str), []).append(stats["entropy_bits"])
        entropy_mean = {k: sum(v) / len(v) for k, v in sorted(entropy.items())}

        print(f"[{ts()}] [step 3/3] evaluating split scenarios", flush=True)

        def scenario(keep_each_end: int) -> dict:
            """Middle layers at IQ2_XXS, first/last keep_each_end at Q4_K_M."""
            q4_layers = list(range(keep_each_end)) + list(range(N_LAYERS - keep_each_end, N_LAYERS))
            iq2_layers = [l for l in range(N_LAYERS) if l not in q4_layers]
            all_cpu = TOP_K / N_EXPERTS * (sum(q4[l] for l in q4_layers)
                                           + sum(iq2[l] for l in iq2_layers))
            # Capacity-optimal placement: every expert layer reads top-k per token,
            # so bytes/token-offloaded per VRAM byte is uniform across layers of a
            # given quant; fill the budget largest-first (Q4 before IQ2).
            budget = VRAM_EXPERT_BUDGET_GB * 1e9
            gpu_bytes = 0
            for l in sorted(q4_layers, key=lambda x: -q4[x]):
                if q4[l] <= budget:
                    budget -= q4[l]; gpu_bytes += q4[l]
            for l in sorted(iq2_layers, key=lambda x: -iq2[x]):
                if iq2[l] <= budget:
                    budget -= iq2[l]; gpu_bytes += iq2[l]
            cpu_bytes = all_cpu - TOP_K / N_EXPERTS * gpu_bytes
            tg_roof = DDR4_READ_GBS * 1e9 / cpu_bytes * ROOFLINE_CALIBRATION
            return {
                "keep_q4_each_end": keep_each_end,
                "iq2_layers": len(iq2_layers),
                "expert_gb_file": round((sum(q4[l] for l in q4_layers)
                                         + sum(iq2[l] for l in iq2_layers)) / 1e9, 2),
                "mib_per_token_all_cpu": round(all_cpu / 2**20, 1),
                "mib_per_token_at_optimum": round(cpu_bytes / 2**20, 1),
                "tg_roofline_calibrated": round(tg_roof, 1),
                "tg_vs_baseline": round(tg_roof / BASELINE_TG, 3),
            }

        # Baseline: all Q4_K_M experts (measured Track-1 optimum placement).
        base_all_cpu = TOP_K / N_EXPERTS * sum(q4.values())
        budget = VRAM_EXPERT_BUDGET_GB * 1e9
        base_gpu = 0
        for l in sorted(q4, key=lambda x: -q4[x]):
            if q4[l] <= budget:
                budget -= q4[l]; base_gpu += q4[l]
        base_cpu_bytes = base_all_cpu - TOP_K / N_EXPERTS * base_gpu
        BASELINE_TG = DDR4_READ_GBS * 1e9 / base_cpu_bytes * ROOFLINE_CALIBRATION

        scenarios = [scenario(k) for k in (12, 8, 4, 0)]
        baseline = {
            "mib_per_token_all_cpu": round(base_all_cpu / 2**20, 1),
            "gpu_expert_gb": round(base_gpu / 1e9, 2),
            "mib_per_token_at_optimum": round(base_cpu_bytes / 2**20, 1),
            "tg_roofline_calibrated": round(BASELINE_TG, 1),
            "note": "measured Track-1 optimum: 33-35 tok/s (docs/moe-track-plan.md)",
        }
    except KeyboardInterrupt:
        OUT.write_text(json.dumps(
            build_payload(q4, iq2, entropy_mean, scenarios, baseline,
                          status="interrupted"), indent=1), encoding="utf-8")
        print(f"\n[{ts()}] interrupted — partial results -> {OUT}", flush=True)
        sys.exit(130)

    OUT.write_text(json.dumps(
        build_payload(q4, iq2, entropy_mean, scenarios, baseline,
                      status="complete"), indent=1), encoding="utf-8")
    print(f"[{ts()}] wrote {OUT}", flush=True)
    print(f"baseline tg prediction {BASELINE_TG:.1f} (measured 33-35)",
          flush=True)
    for s in scenarios:
        print(f"keep {s['keep_q4_each_end']:2d}/end: {s['mib_per_token_at_optimum']:7.1f} MiB/tok "
              f"-> {s['tg_roofline_calibrated']:5.1f} tok/s ({s['tg_vs_baseline']:.2f}x)",
              flush=True)


if __name__ == "__main__":
    main()
