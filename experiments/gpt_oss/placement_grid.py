"""gpt-oss-20b MXFP4 placement grid: second-architecture roofline validation.

Track-1 established the Qwen3-30B-A3B placement optimum and the DDR4 roofline
model (generation tok/s ~ bandwidth / expert-bytes-per-token). This grid
re-measures that roofline on a second, independently architected MoE:
gpt-oss-20b (24 layers, 32 experts, top-4 — trace-verified, 21B total / 3.6B active, native
MXFP4 = Q4-class active path, 12.1 GB GGUF). If the model holds, the
all-experts-CPU point (24, 24) lands near bandwidth/active-bytes and the
curve bends upward as experts move into VRAM, exactly as Qwen3 did.

Same protocol as the Track-1 grid: unmodified b10355 llama-bench, 128 prompt
tokens, 32 generated, 3 repetitions, mmap, f16 KV. Failures (e.g. OOM at
(24, 0): 12.1 GB cannot fit 8 GiB VRAM) are recorded, never dropped.

Smoke result already on record (stock b10355 llama-completion, default --fit,
47 generated tokens): 29.52 tok/s, coherent output — MXFP4 on sm_86 works.

Child stdout/stderr stream live by default (--quiet suppresses). Ctrl+C
terminates the running child, writes partial results to the output JSON with
status="interrupted", and exits with code 130.

Output: results/gpt-oss/placement-grid.json (placement-grid-refine.json with
--refine)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "system"))
from common import run_live, ts  # noqa: E402

BENCH = ROOT / "tools/llama.cpp-b10355/llama-bench.exe"
MODEL = ROOT / "models/gpt-oss-20b-GGUF/gpt-oss-20b-MXFP4.gguf"
OUT = ROOT / "results/gpt-oss/placement-grid.json"

# (ngl, n_cpu_moe). gpt-oss-20b has 24 layers. (24, 24) is the
# gpt-oss-120b-analog regime: attention GPU, all experts RAM-resident.
GRID = [
    (24, 24),  # all attention GPU, all experts CPU  <- roofline point
    (24, 16),
    (24, 12),
    (24, 8),
    (24, 4),
    (24, 0),   # full offload attempt — expected OOM (capacity boundary record)
    (0, 0),    # pure CPU floor
]

# First pass peaked at K=12 (42.62 +/- 2.22). Refinement brackets the peak
# with 5 reps, same as the Track-1 refinement protocol.
REFINE = [
    (24, 14),
    (24, 12),
    (24, 10),
]


def write_payload(out: Path, grid: list, reps: str, records: list[dict],
                  status: str) -> None:
    payload = {
        "status": status,
        "bench": "llama-bench b10355 (unmodified)",
        "model": MODEL.name,
        "model_facts": "gpt-oss-20b: 24 layers, 32 experts (trace-verified), top-4, 21B total / 3.6B active, native MXFP4, 13.22 MB/expert -> 1.27 GB expert bytes/token",
        "protocol": f"128 prompt tokens, 32 generated, {reps} repetitions, mmap, f16 KV",
        "references": {
            "qwen3_30b_a3b_best_tok_s": "33-35 (-ngl 48 --n-cpu-moe 33, docs/moe-track-plan.md)",
            "smoke_default_fit_tok_s": 29.52,
            "roofline_prediction_all_cpu_tok_s": "~26.6 GB/s measured effective DDR4 read (B07 + implied 21.0 tok/s x 1.27 GB/token) -> all-experts-CPU ceiling ~21 tok/s; 120b (1.9 GB/token) ceiling ~14 tok/s",
        },
        "grid": [{"ngl": n, "n_cpu_moe": k} for n, k in grid],
        "records": records,
    }
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--refine", action="store_true",
                    help="run the 3-point refinement grid (5 reps) instead of "
                         "the full grid")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress live child output (banners still print)")
    args = ap.parse_args()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    grid = REFINE if args.refine else GRID
    reps = "5" if args.refine else "3"
    out = OUT.parent / "placement-grid-refine.json" if args.refine else OUT
    records: list[dict] = []
    mode = "refinement" if args.refine else "full"
    print(f"[{ts()}] placement grid ({mode}): {len(grid)} (ngl, n-cpu-moe) "
          f"points, {reps} reps, model {MODEL.name}, bench {BENCH.name}",
          flush=True)
    print(f"[{ts()}] output -> {out} (Ctrl+C writes partial results)",
          flush=True)
    try:
        for i, (ngl, ncpu) in enumerate(grid, 1):
            label = f"[step {i}/{len(grid)}] ngl={ngl} n-cpu-moe={ncpu}"
            cmd = [
                str(BENCH), "-m", str(MODEL),
                "-ngl", str(ngl), "--n-cpu-moe", str(ncpu),
                "-p", "128", "-n", "32", "-r", reps, "-o", "json",
            ]
            print(f"[{ts()}] {label}: measuring", flush=True)
            res = run_live(cmd, label, quiet=args.quiet, timeout=1800)
            record = {"ngl": ngl, "n_cpu_moe": ncpu, "rc": res["rc"],
                      "wall_s": res["wall_s"]}
            if res["rc"] == 0:
                try:
                    rows = json.loads(res["stdout"])
                    record["rows"] = [
                        {"test": r.get("test"), "n_prompt": r.get("n_prompt"),
                         "n_gen": r.get("n_gen"), "avg_ns": r.get("avg_ns"),
                         "stddev_ns": r.get("stddev_ns"), "avg_ts": r.get("avg_ts"),
                         "stddev_ts": r.get("stddev_ts")}
                        for r in rows
                    ]
                    gen = [r["avg_ts"] for r in record["rows"] if r.get("n_gen")]
                    pp = [r["avg_ts"] for r in record["rows"] if r.get("n_prompt")]
                    print(f"[{ts()}] {label}: ok pp={pp[0]:.1f} "
                          f"tg={gen[0]:.2f} tok/s", flush=True)
                except json.JSONDecodeError:
                    record["status"] = "parse_error"
                    record["stdout_head"] = res["stdout"][:300]
            else:
                record["status"] = "error"
                record["stderr_tail"] = res["stderr"][-400:]
                print(f"[{ts()}] {label}: FAILED rc={res['rc']}: "
                      f"{res['stderr'][-200:]!r}", flush=True)
            records.append(record)
    except KeyboardInterrupt:
        write_payload(out, grid, reps, records, status="interrupted")
        print(f"\n[{ts()}] interrupted — partial results ({len(records)} "
              f"rows) -> {out}", flush=True)
        sys.exit(130)

    write_payload(out, grid, reps, records, status="complete")
    print(f"\n[{ts()}] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
