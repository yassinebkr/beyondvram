"""Track 2 baseline: placement sweep for dense over-VRAM Qwen3-32B Q4_K_M.

Question: how well does unmodified llama.cpp run a dense 32B model from RAM
with partial GPU offload on the RTX 3070 Ti (8 GiB), and where is the
placement optimum? Mirrors the Track-1 MoE baseline protocol so the two are
comparable: llama-bench b10355, 128 prompt / 32 gen tokens, 3 repetitions,
mmap, f16 KV. Failures (OOM) are recorded, never dropped.
Output: results/track2-dense/placement-sweep-qwen3-32b.json
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "tools/llama.cpp-b10355/llama-bench.exe"
MODEL = ROOT / "models/Qwen3-32B-GGUF/Qwen3-32B-Q4_K_M.gguf"
OUT = ROOT / "results/track2-dense/placement-sweep-qwen3-32b.json"

# Qwen3-32B: 64 layers, ~310 MB/layer at Q4_K_M. ~7 GiB usable VRAM ->
# roughly 20-22 layers plus KV/context fit; sweep brackets that region and
# records OOM failures honestly.
NGL_VALUES = [0, 8, 16, 20, 22, 24, 32]


def main() -> None:
    if not MODEL.exists():
        raise SystemExit(f"model not found: {MODEL}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for ngl in NGL_VALUES:
        cmd = [str(BENCH), "-m", str(MODEL), "-ngl", str(ngl),
               "-p", "128", "-n", "32", "-r", "3", "-o", "json"]
        print(f"[sweep] ngl={ngl}", flush=True)
        start = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              stdin=subprocess.DEVNULL, timeout=1800)
        wall = time.perf_counter() - start
        record = {"ngl": ngl, "rc": proc.returncode, "wall_s": round(wall, 1)}
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
        "model": MODEL.name,
        "protocol": "128 prompt tokens, 32 generated, 3 repetitions, mmap, f16 KV",
        "references": {
            "moe_qwen3_30b_a3b_best": {"config": "-ngl 48 --n-cpu-moe 33",
                                       "tok_s": [34.71, 3.04]},
        },
        "records": records,
    }
    OUT.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
