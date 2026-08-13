"""Performance measurement for the MoE expert-cache PoC.

Runs llama-moe-cache.exe over several cache configurations (including cache-off
as the internal baseline), 3 repetitions x 128 generated tokens each, greedy,
fixed seed and prompt set. Parses llama.cpp eval timing from stderr for tok/s.
Every repetition is kept; failures are recorded as status entries, never dropped.
Output: results/moe-locality/moe-cache-poc.json
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXE = ROOT / "tools/llama.cpp-source/build/bin/llama-moe-cache.exe"
MODEL = ROOT / "models/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf"
OUT_DIR = ROOT / "results/moe-locality"

CONFIGS = [
    ("off", None),
    ("8:0:8", "8:0:8"),
    ("8:0:23", "8:0:23"),
]
PROMPTS = [
    "Write a short introduction to the history of the printing press.",
    "Explain step by step why the square root of 2 is irrational.",
    "Tell a short story about a lighthouse keeper who finds a strange letter.",
]
N_PREDICT = 128
REPS = 3

GEN_RE = re.compile(r"gen time = [\d.]+ ms / (\d+) tokens / ([\d.]+) tokens per second")


def parse_tok_s(stderr: str) -> tuple[float | None, int | None]:
    for line in reversed(stderr.splitlines()):
        match = GEN_RE.search(line)
        if match:
            return float(match.group(2)), int(match.group(1))
    return None, None


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for config_name, cache_env in CONFIGS:
        for prompt in PROMPTS:
            for rep in range(REPS):
                env = dict(os.environ)
                env["GGML_CUDA_DISABLE_GRAPHS"] = "1"
                stats_path = OUT_DIR / f"cache-stats-{config_name.replace(':','-')}-r{rep}.json"
                if cache_env:
                    env["LLAMA_MOE_CACHE"] = cache_env
                    env["MOE_CACHE_STATS_OUT"] = str(stats_path)
                else:
                    env.pop("LLAMA_MOE_CACHE", None)
                cmd = [
                    str(EXE), "-m", str(MODEL), "-p", prompt,
                    "-n", str(N_PREDICT), "-ngl", "18",
                    "--seed", "42", "--temp", "0", "--no-mmap",
                ]
                tag = f"{config_name} | {prompt[:24]}... | rep {rep}"
                print(f"[run] {tag}", flush=True)
                start = time.perf_counter()
                proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                                      stdin=subprocess.DEVNULL, timeout=1200)
                wall = time.perf_counter() - start
                tok_s, n_runs = parse_tok_s(proc.stderr)
                record = {
                    "config": config_name, "prompt": prompt, "rep": rep,
                    "rc": proc.returncode, "wall_s": round(wall, 1),
                    "tok_per_s": tok_s, "eval_runs": n_runs,
                    "status": "ok" if proc.returncode == 0 else "error",
                }
                if proc.returncode != 0:
                    record["stderr_tail"] = proc.stderr[-500:]
                records.append(record)
                print(f"  rc={proc.returncode} tok/s={tok_s} wall={wall:.0f}s", flush=True)
                (OUT_DIR / f"poc-{config_name.replace(':','-')}-r{rep}.txt").write_text(
                    proc.stdout, encoding="utf-8")

    payload = {
        "exe": "llama-moe-cache (pinned dd1ea52 + moe-cache module)",
        "model": MODEL.name, "n_predict": N_PREDICT, "reps": REPS,
        "prompts": PROMPTS, "configs": [c for c, _ in CONFIGS],
        "notes": [
            "cache-off uses the same binary with LLAMA_MOE_CACHE unset (internal baseline)",
            "stock b10355 baseline reference: 25.15 +/- 2.40 tok/s at -ngl 18",
            "--n-cpu-moe 48 ablation reference: 16.87 +/- 0.80 tok/s",
        ],
        "records": records,
    }
    out = OUT_DIR / "moe-cache-poc.json"
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")

    # quick summary
    from statistics import mean, stdev
    for config_name, _ in CONFIGS:
        vals = [r["tok_per_s"] for r in records
                if r["config"] == config_name and r["tok_per_s"] is not None]
        if vals:
            print(f"{config_name:>8}: {mean(vals):.2f} +/- {stdev(vals):.2f} tok/s (n={len(vals)})")


if __name__ == "__main__":
    main()
