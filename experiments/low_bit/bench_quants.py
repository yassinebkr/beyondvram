"""Track 3: generation rate and placement for low-bit model candidates.

Candidates (all local GGUFs) against the measured anchors:
- Qwen3-32B Q4_K_M (anchor, Track 2: ~2.7 tok/s best)
- Qwen3-32B Q3_K_M and IQ2_XXS (bartowski, community quants)
- microsoft BitNet b1.58 2B 4T, i2_s GGUF (ternary, VRAM-resident)
- Qwen3-30B-A3B MoE (anchor, Track 1: ~33-35 tok/s best)

Protocol mirrors Tracks 1/2: llama-bench b10355, 128 prompt / 32 gen, 3 reps,
mmap, f16 KV. Failures recorded, never dropped.
Output: results/track3-low-bit/bench-quants.json
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "tools/llama.cpp-b10355/llama-bench.exe"
OUT = ROOT / "results/track3-low-bit/bench-quants.json"

MODELS = [
    # (id, path, ngl values to sweep)
    ("qwen3-32b-q3km", "models/Qwen3-32B-GGUF/Qwen_Qwen3-32B-Q3_K_M.gguf", [0, 22, 28, 40]),
    ("qwen3-32b-iq2xxs", "models/Qwen3-32B-GGUF/Qwen_Qwen3-32B-IQ2_XXS.gguf", [22, 40, 56, 64]),
    ("bitnet-2b-i2s", "models/bitnet-b1.58-2B-4T/ggml-model-i2_s.gguf", [0, 99]),
]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for model_id, rel_path, ngls in MODELS:
        model = ROOT / rel_path
        if not model.exists():
            records.append({"model": model_id, "status": "missing",
                            "path": rel_path})
            print(f"[skip] {model_id}: not found", flush=True)
            continue
        size_gb = model.stat().st_size / 1e9
        for ngl in ngls:
            cmd = [str(BENCH), "-m", str(model), "-ngl", str(ngl),
                   "-p", "128", "-n", "32", "-r", "3", "-o", "json"]
            print(f"[bench] {model_id} ngl={ngl}", flush=True)
            start = time.perf_counter()
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  stdin=subprocess.DEVNULL, timeout=1800)
            record = {"model": model_id, "size_gb": round(size_gb, 2),
                      "ngl": ngl, "rc": proc.returncode,
                      "wall_s": round(time.perf_counter() - start, 1)}
            if proc.returncode == 0:
                try:
                    rows = json.loads(proc.stdout)
                    record["rows"] = [
                        {"n_prompt": r.get("n_prompt"), "n_gen": r.get("n_gen"),
                         "avg_ts": r.get("avg_ts"), "stddev_ts": r.get("stddev_ts")}
                        for r in rows
                    ]
                    gen = [r["avg_ts"] for r in record["rows"] if r.get("n_gen")]
                    pp = [r["avg_ts"] for r in record["rows"] if r.get("n_prompt")]
                    print(f"  ok: pp={pp[0]:.1f} tg={gen[0]:.2f} tok/s", flush=True)
                except json.JSONDecodeError:
                    record["status"] = "parse_error"
                    record["stdout_head"] = proc.stdout[:300]
            else:
                record["status"] = "error"
                record["stderr_tail"] = proc.stderr[-400:]
                print(f"  FAILED rc={proc.returncode}: {proc.stderr[-200:]!r}", flush=True)
            records.append(record)

    payload = {
        "bench": "llama-bench b10355 (unmodified)",
        "protocol": "128 prompt tokens, 32 generated, 3 repetitions, mmap, f16 KV",
        "anchors": {
            "qwen3-32b-q4km_best": {"ngl": 22, "tok_s": 2.66},
            "qwen3-30b-a3b-q4km_best": {"config": "-ngl 48 --n-cpu-moe 33",
                                        "tok_s": 34.71},
        },
        "records": records,
    }
    OUT.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
