"""Track 3: generation rate and placement for low-bit model candidates.

Candidates (all local GGUFs) against the measured anchors:
- Qwen3-32B Q4_K_M (anchor, Track 2: ~2.7 tok/s best)
- Qwen3-32B Q3_K_M and IQ2_XXS (bartowski, community quants)
- microsoft BitNet b1.58 2B 4T, i2_s GGUF (ternary, VRAM-resident)
- Qwen3-30B-A3B MoE (anchor, Track 1: ~33-35 tok/s best)

Protocol mirrors Tracks 1/2: llama-bench b10355, 128 prompt / 32 gen, 3 reps,
mmap, f16 KV. Failures recorded, never dropped.

Child stdout/stderr stream live by default (--quiet suppresses). Ctrl+C
terminates the running child, writes partial results to the output JSON with
status="interrupted", and exits with code 130.

Output: results/track3-low-bit/bench-quants.json
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
OUT = ROOT / "results/track3-low-bit/bench-quants.json"

MODELS = [
    # (id, path, ngl values to sweep)
    ("qwen3-32b-q3km", "models/Qwen3-32B-GGUF/Qwen_Qwen3-32B-Q3_K_M.gguf", [0, 22, 28, 40]),
    ("qwen3-32b-iq2xxs", "models/Qwen3-32B-GGUF/Qwen_Qwen3-32B-IQ2_XXS.gguf", [22, 40, 56, 64]),
    ("bitnet-2b-i2s", "models/bitnet-b1.58-2B-4T/ggml-model-i2_s.gguf", [0, 99]),
]


def write_payload(records: list[dict], status: str) -> None:
    payload = {
        "status": status,
        "bench": "llama-bench b10355 (unmodified)",
        "protocol": "128 prompt tokens, 32 generated, 3 repetitions, mmap, f16 KV",
        "anchors": {
            "qwen3-32b-q4km_best": {"ngl": 22, "tok_s": 2.66},
            "qwen3-30b-a3b-q4km_best": {"config": "-ngl 48 --n-cpu-moe 33",
                                        "tok_s": 34.71},
        },
        "records": records,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quiet", action="store_true",
                    help="suppress live child output (banners still print)")
    args = ap.parse_args()

    total = sum(len(ngls) for _, rel, ngls in MODELS if (ROOT / rel).exists())
    records: list[dict] = []
    print(f"[{ts()}] low-bit bench: {len(MODELS)} candidate models, "
          f"{total} bench configs (128 prompt / 32 gen / 3 reps, mmap, f16 KV)",
          flush=True)
    print(f"[{ts()}] output -> {OUT} (Ctrl+C writes partial results)",
          flush=True)
    step = 0
    try:
        for model_id, rel_path, ngls in MODELS:
            model = ROOT / rel_path
            if not model.exists():
                records.append({"model": model_id, "status": "missing",
                                "path": rel_path})
                print(f"[{ts()}] {model_id}: SKIPPED (not found)", flush=True)
                continue
            size_gb = model.stat().st_size / 1e9
            for ngl in ngls:
                step += 1
                label = f"[{step}/{total}] {model_id} ngl={ngl}"
                cmd = [str(BENCH), "-m", str(model), "-ngl", str(ngl),
                       "-p", "128", "-n", "32", "-r", "3", "-o", "json"]
                print(f"[{ts()}] [step {step}/{total}] {model_id} ngl={ngl}: "
                      f"measuring", flush=True)
                res = run_live(cmd, label, quiet=args.quiet, timeout=1800)
                record = {"model": model_id, "size_gb": round(size_gb, 2),
                          "ngl": ngl, "rc": res["rc"], "wall_s": res["wall_s"]}
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
