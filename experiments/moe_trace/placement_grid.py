"""Decoupled placement grid: -ngl (attention+experts of last N layers) x --n-cpu-moe K.

The original Track-1 sweep varied only -ngl. This grid decouples attention
placement from expert placement: -ngl offloads the LAST N layers whole, and
--n-cpu-moe K forces the FIRST K layers' expert tensors back to CPU. Of
particular interest: all-48-layers attention on GPU (attention is ~12
MB/layer and always fits) with only a suffix of layers' experts in VRAM.

Uses the unmodified b10355 llama-bench (same protocol as the recorded
baseline sweep: 128 prompt tokens, 32 generated, 3 repetitions, mmap, f16 KV).
Failures (e.g. OOM) are recorded as status entries, never dropped.

Child stdout/stderr stream live by default (--quiet suppresses). Ctrl+C
terminates the running child, writes partial results to the output JSON with
status="interrupted", and exits with code 130.

Output: results/moe-locality/placement-grid.json
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
MODEL = ROOT / "models/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf"
OUT = ROOT / "results/moe-locality/placement-grid.json"

# (ngl, n_cpu_moe): interior of the two-knob space. (18, 0) is the recorded
# 25.15 tok/s baseline; (18, 48) is the recorded 16.87 ablation.
GRID = [
    (48, 48),  # all attention GPU, all experts CPU
    (48, 40),  # all attention GPU, experts of layers 40-47 in VRAM
    (48, 36),
    (48, 34),  # ~14 layers of experts in VRAM (~5 GB) — expected VRAM limit
    (48, 30),
    (36, 30),
    (36, 24),
    (30, 24),
    (24, 24),
    (24, 18),
]


def write_payload(records: list[dict], status: str) -> None:
    payload = {
        "status": status,
        "bench": "llama-bench b10355 (unmodified)",
        "model": MODEL.name,
        "protocol": "128 prompt tokens, 32 generated, 3 repetitions, mmap, f16 KV",
        "references": {
            "ngl18_baseline_tok_s": [25.15, 2.40],
            "ngl18_cpu_moe48_tok_s": [16.87, 0.80],
        },
        "grid": [{"ngl": n, "n_cpu_moe": k} for n, k in GRID],
        "records": records,
    }
    OUT.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quiet", action="store_true",
                    help="suppress live child output (banners still print)")
    args = ap.parse_args()

    records: list[dict] = []
    print(f"[{ts()}] placement grid: {len(GRID)} (ngl, n-cpu-moe) points, "
          f"model {MODEL.name}, bench {BENCH.name}", flush=True)
    print(f"[{ts()}] output -> {OUT} (Ctrl+C writes partial results)",
          flush=True)
    try:
        for i, (ngl, ncpu) in enumerate(GRID, 1):
            label = f"[step {i}/{len(GRID)}] ngl={ngl} n-cpu-moe={ncpu}"
            cmd = [
                str(BENCH), "-m", str(MODEL),
                "-ngl", str(ngl), "--n-cpu-moe", str(ncpu),
                "-p", "128", "-n", "32", "-r", "3", "-o", "json",
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
        write_payload(records, status="interrupted")
        print(f"\n[{ts()}] interrupted — partial results ({len(records)} "
              f"rows) -> {OUT}", flush=True)
        sys.exit(130)

    write_payload(records, status="complete")
    print(f"\n[{ts()}] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
