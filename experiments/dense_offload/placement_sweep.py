"""Track 2 baseline: placement sweep for dense over-VRAM Qwen3-32B Q4_K_M.

Question: how well does unmodified llama.cpp run a dense 32B model from RAM
with partial GPU offload on the RTX 3070 Ti (8 GiB), and where is the
placement optimum? Mirrors the Track-1 MoE baseline protocol so the two are
comparable: llama-bench b10355, 128 prompt / 32 gen tokens, 3 repetitions,
mmap, f16 KV. Failures (OOM) are recorded, never dropped.

Child stdout/stderr stream live by default (--quiet suppresses). Ctrl+C
terminates the running child, writes partial results to the output JSON with
status="interrupted", and exits with code 130.

Output: results/track2-dense/placement-sweep-qwen3-32b.json
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
MODEL = ROOT / "models/Qwen3-32B-GGUF/Qwen3-32B-Q4_K_M.gguf"
OUT = ROOT / "results/track2-dense/placement-sweep-qwen3-32b.json"

# Qwen3-32B: 64 layers, ~310 MB/layer at Q4_K_M. ~7 GiB usable VRAM ->
# roughly 20-22 layers plus KV/context fit; sweep brackets that region and
# records OOM failures honestly.
NGL_VALUES = [0, 8, 16, 20, 22, 24, 32]


def write_payload(records: list[dict], status: str) -> None:
    payload = {
        "status": status,
        "bench": "llama-bench b10355 (unmodified)",
        "model": MODEL.name,
        "protocol": "128 prompt tokens, 32 generated, 3 repetitions, mmap, f16 KV",
        "references": {
            "moe_qwen3_30b_a3b_best": {"config": "-ngl 48 --n-cpu-moe 33",
                                       "tok_s": [34.71, 3.04]},
        },
        "records": records,
    }
    OUT.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quiet", action="store_true",
                    help="suppress live child output (banners still print)")
    args = ap.parse_args()

    if not MODEL.exists():
        raise SystemExit(f"model not found: {MODEL}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    print(f"[{ts()}] placement sweep: {len(NGL_VALUES)} placements, "
          f"model {MODEL.name}, bench {BENCH.name}", flush=True)
    print(f"[{ts()}] output -> {OUT} (Ctrl+C writes partial results)",
          flush=True)
    try:
        for i, ngl in enumerate(NGL_VALUES, 1):
            label = f"[step {i}/{len(NGL_VALUES)}] ngl={ngl}"
            cmd = [str(BENCH), "-m", str(MODEL), "-ngl", str(ngl),
                   "-p", "128", "-n", "32", "-r", "3", "-o", "json"]
            print(f"[{ts()}] {label}: measuring", flush=True)
            res = run_live(cmd, label, quiet=args.quiet, timeout=1800)
            record = {"ngl": ngl, "rc": res["rc"], "wall_s": res["wall_s"]}
            if res["rc"] == 0:
                try:
                    rows = json.loads(res["stdout"])
                    record["rows"] = [
                        {"n_prompt": r.get("n_prompt"), "n_gen": r.get("n_gen"),
                         "avg_ts": r.get("avg_ts"), "stddev_ts": r.get("stddev_ts")}
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
