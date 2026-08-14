"""Performance measurement for the MoE expert-cache PoC.

Runs llama-moe-cache.exe over several cache configurations (including cache-off
as the internal baseline), 3 repetitions x 128 generated tokens each, greedy,
fixed seed and prompt set. Parses llama.cpp eval timing from stderr for tok/s.
Every repetition is kept; failures are recorded as status entries, never dropped.

Child stdout/stderr stream live by default (--quiet suppresses). Ctrl+C
terminates the running child, writes partial results to the output JSON with
status="interrupted", and exits with code 130.

Output: results/moe-locality/moe-cache-poc.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "system"))
from common import run_live, ts  # noqa: E402

EXE = ROOT / "tools/llama.cpp-source/build/bin/llama-moe-cache.exe"
MODEL = ROOT / "models/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf"
OUT_DIR = ROOT / "results/moe-locality"
OUT = OUT_DIR / "moe-cache-poc.json"

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


def write_payload(records: list[dict], status: str) -> None:
    payload = {
        "status": status,
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
    OUT.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quiet", action="store_true",
                    help="suppress live child output (banners still print)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env_base = dict(os.environ)
    total = len(CONFIGS) * len(PROMPTS) * REPS
    print(f"[{ts()}] cache PoC: {len(CONFIGS)} configs x {len(PROMPTS)} prompts "
          f"x {REPS} reps = {total} runs, {N_PREDICT} tokens each, -ngl 18",
          flush=True)
    print(f"[{ts()}] output -> {OUT} (Ctrl+C writes partial results)",
          flush=True)

    records = []
    step = 0
    try:
        for config_name, cache_env in CONFIGS:
            for prompt in PROMPTS:
                for rep in range(REPS):
                    step += 1
                    # run_live inherits the process environment; rebuild it
                    # from the startup snapshot each run so the child sees the
                    # same env the old env=dict(os.environ)+overrides
                    # construction produced (LLAMA_MOE_CACHE absent for "off").
                    os.environ.clear()
                    os.environ.update(env_base)
                    os.environ["GGML_CUDA_DISABLE_GRAPHS"] = "1"
                    stats_path = OUT_DIR / f"cache-stats-{config_name.replace(':','-')}-r{rep}.json"
                    if cache_env:
                        os.environ["LLAMA_MOE_CACHE"] = cache_env
                        os.environ["MOE_CACHE_STATS_OUT"] = str(stats_path)
                    else:
                        os.environ.pop("LLAMA_MOE_CACHE", None)
                    cmd = [
                        str(EXE), "-m", str(MODEL), "-p", prompt,
                        "-n", str(N_PREDICT), "-ngl", "18",
                        "--seed", "42", "--temp", "0", "--no-mmap",
                    ]
                    tag = f"{config_name} | {prompt[:24]}... | rep {rep}"
                    label = f"[step {step}/{total}] {tag}"
                    print(f"[{ts()}] {label}", flush=True)
                    res = run_live(cmd, label, quiet=args.quiet, timeout=1200)
                    tok_s, n_runs = parse_tok_s(res["stderr"])
                    record = {
                        "config": config_name, "prompt": prompt, "rep": rep,
                        "rc": res["rc"], "wall_s": res["wall_s"],
                        "tok_per_s": tok_s, "eval_runs": n_runs,
                        "status": "ok" if res["rc"] == 0 else "error",
                    }
                    if res["rc"] != 0:
                        record["stderr_tail"] = res["stderr"][-500:]
                    records.append(record)
                    print(f"[{ts()}] {label} rc={res['rc']} tok/s={tok_s} "
                          f"wall={res['wall_s']}s", flush=True)
                    (OUT_DIR / f"poc-{config_name.replace(':','-')}-r{rep}.txt").write_text(
                        res["stdout"], encoding="utf-8")
    except KeyboardInterrupt:
        write_payload(records, status="interrupted")
        print(f"\n[{ts()}] interrupted — partial results ({len(records)} "
              f"rows) -> {OUT}", flush=True)
        sys.exit(130)

    write_payload(records, status="complete")
    print(f"\n[{ts()}] wrote {OUT}", flush=True)

    # quick summary
    from statistics import mean, stdev
    for config_name, _ in CONFIGS:
        vals = [r["tok_per_s"] for r in records
                if r["config"] == config_name and r["tok_per_s"] is not None]
        if vals:
            print(f"{config_name:>8}: {mean(vals):.2f} +/- {stdev(vals):.2f} tok/s (n={len(vals)})",
                  flush=True)


if __name__ == "__main__":
    main()
