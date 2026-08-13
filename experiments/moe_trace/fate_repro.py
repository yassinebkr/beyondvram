"""Reproduce-or-falsify measurement for the FATE fork (ongunm/llama-moe-cache).

Third-party AGPL-3.0 fork claiming Qwen3-30B-A3B Q4_K_M 33.74 -> 64.45 tok/s
via a GPU expert pool + cross-layer/temporal prefetch. This script runs the
fork's llama-completion with --fate off (internal baseline) and --fate on at
several --fate-cache pool sizes, 3 repetitions x 128 generated tokens, greedy,
fixed seed and prompt set. Parses eval timing and the FATE cache stats from
stderr. Every repetition is kept; failures are recorded, never dropped.

Parity gate is NOT in this script: output equivalence (fork --fate off vs
stock b10355, and --fate on vs off, token-exact at temp 0) is checked first
with direct runs; do not trust speed numbers before parity passes.

Output: results/moe-locality/fate-repro/fate-repro.json + per-rep raw logs.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXE = ROOT / "tools/llama-moe-cache/build/bin/llama-completion.exe"
MODEL = ROOT / "models/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf"
OUT_DIR = ROOT / "results/moe-locality/fate-repro"

# Pool sizes bracket the fork author's 2048 MB (1.44x working set on his
# 12 GB card) down to below the 1418 MB per-token working set, where LRU
# churn must collapse the hit rate if the mechanism is real.
CONFIGS = [
    ("off", None),
    ("2048", "2048"),
    ("1536", "1536"),
    ("1024", "1024"),
]
PROMPTS = [
    "Write a short introduction to the history of the printing press.",
    "Explain step by step why the square root of 2 is irrational.",
    "Tell a short story about a lighthouse keeper who finds a strange letter.",
]
N_PREDICT = 128
REPS = 3

# mmap is REQUIRED: with --no-mmap the fork's cudaHostRegister partially pins
# malloc'd expert weights, enabling truly async H2D into reused input_cpy
# buffers on this 8 GB placement -> CUDA invalid argument (fork bug, absent on
# the author's 12 GB card where no buffer reuse occurs). With mmap, 0/144
# tensors pin, H2D is pageable/serializing, and runs complete.
# FATE_NO_STAGING=1 is REQUIRED for correctness: the fork's single shared
# staging buffer races with its own in-flight async H2D (slots receive
# whatever staging holds at DMA time -> garbage tokens from token 1, as
# shipped). Direct pageable H2D from mmap'd weights is synchronizing but
# correct; with it, --fate output is token-exact vs --fate off (parity2 logs).
# The earlier 98.96% hit rate under the race was an artifact: corrupted runs
# emit one repeated token, making temporal prediction trivially perfect.
# Honest reference points from the parity runs: 56.46% hit at 1663 slots on
# real text, 3.98 t/s vs 27.67 t/s off (95-token evals).
EXTRA_ENV: dict[str, str] = {"FATE_NO_STAGING": "1"}

EVAL_RE = re.compile(r"^\s*common_perf_print:\s+eval time\s+=\s*[\d.]+\s*ms\s+/\s+\d+ runs\s+\(\s*[\d.]+\s*ms per token,\s*([\d.]+) tokens per second")
PP_RE = re.compile(r"prompt eval time\s+=\s*[\d.]+\s*ms\s+/\s+\d+ tokens\s+\(\s*[\d.]+\s*ms per token,\s*([\d.]+) tokens per second")
HIT_RE = re.compile(r"hit rate\s+:\s+([\d.]+)%")
PREFETCH_RE = re.compile(r"prefetched\s+:\s+(\d+)")
POOL_RE = re.compile(r"pool\s+:\s+(\d+) slots")


def parse_metrics(stderr: str) -> dict:
    tok_s = None
    for line in reversed(stderr.splitlines()):
        match = EVAL_RE.search(line)
        if match:
            tok_s = float(match.group(1))
            break
    pp = PP_RE.search(stderr)
    hit = HIT_RE.search(stderr)
    prefetch = PREFETCH_RE.search(stderr)
    pool = POOL_RE.search(stderr)
    return {
        "tok_per_s": tok_s,
        "prompt_tok_per_s": float(pp.group(1)) if pp else None,
        "fate_hit_rate_pct": float(hit.group(1)) if hit else None,
        "fate_prefetched": int(prefetch.group(1)) if prefetch else None,
        "fate_pool_slots": int(pool.group(1)) if pool else None,
    }


def summarize(records: list[dict]) -> None:
    from statistics import mean, stdev
    for config_name, _ in CONFIGS:
        vals = [r["tok_per_s"] for r in records
                if r["config"] == config_name and r["tok_per_s"] is not None]
        hits = [r["fate_hit_rate_pct"] for r in records
                if r["config"] == config_name and r["fate_hit_rate_pct"] is not None]
        if vals:
            hit_str = f", hit {mean(hits):.2f}%" if hits else ""
            print(f"{config_name:>6}: {mean(vals):.2f} +/- {stdev(vals):.2f} tok/s (n={len(vals)}){hit_str}")


def reparse() -> None:
    """Rebuild fate-repro.json from the saved per-rep raw logs.

    Parsing is deterministic post-processing; the 12-run GPU matrix is not
    re-executed. Raw .stderr.txt logs remain the source of truth.
    """
    path = OUT_DIR / "fate-repro.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    for record in payload["records"]:
        stem = f"fate-{record['config']}-p{PROMPTS.index(record['prompt'])}-r{record['rep']}"
        stderr_path = OUT_DIR / f"{stem}.stderr.txt"
        if not stderr_path.exists():
            print(f"[reparse] missing {stderr_path.name}")
            continue
        record.update(parse_metrics(stderr_path.read_text(encoding="utf-8", errors="replace")))
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"reparsed {path}")
    summarize(payload["records"])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if "--reparse" in sys.argv:
        reparse()
        return
    records = []
    for config_name, cache_mb in CONFIGS:
        for prompt in PROMPTS:
            for rep in range(REPS):
                cmd = [
                    str(EXE), "-m", str(MODEL), "-p", prompt,
                    "-n", str(N_PREDICT), "--seed", "42", "--temp", "0",
                    "-no-cnv",
                ]
                if cache_mb is not None:
                    cmd += ["--fate", "--fate-cache", cache_mb]
                tag = f"{config_name} | {prompt[:24]}... | rep {rep}"
                print(f"[run] {tag}", flush=True)
                start = time.perf_counter()
                env = {**os.environ, **EXTRA_ENV}
                proc = subprocess.run(cmd, capture_output=True, text=True,
                                      stdin=subprocess.DEVNULL, timeout=1800, env=env)
                wall = time.perf_counter() - start
                record = {
                    "config": config_name, "prompt": prompt, "rep": rep,
                    "rc": proc.returncode, "wall_s": round(wall, 1),
                    "status": "ok" if proc.returncode == 0 else "error",
                    **parse_metrics(proc.stderr),
                }
                if proc.returncode != 0:
                    record["stderr_tail"] = proc.stderr[-800:]
                records.append(record)
                print(f"  rc={proc.returncode} tok/s={record['tok_per_s']} "
                      f"hit={record['fate_hit_rate_pct']} wall={wall:.0f}s", flush=True)
                stem = f"fate-{config_name}-p{PROMPTS.index(prompt)}-r{rep}"
                (OUT_DIR / f"{stem}.out.txt").write_text(proc.stdout, encoding="utf-8")
                (OUT_DIR / f"{stem}.stderr.txt").write_text(proc.stderr, encoding="utf-8")

    payload = {
        "exe": "llama-completion from ongunm/llama-moe-cache (AGPL-3.0, depth-1 clone 2026-08-13)",
        "claim_under_test": "Qwen3-30B-A3B Q4_K_M 33.74 -> 64.45 tok/s, 99.50% hit, RTX 4070 Ti 12 GB",
        "this_machine": "RTX 3070 Ti 8 GiB, Zen 3 6c/12t, 32 GiB DDR4-3600, Windows 11",
        "model": MODEL.name, "n_predict": N_PREDICT, "reps": REPS,
        "prompts": PROMPTS, "configs": [c for c, _ in CONFIGS],
        "notes": [
            "default --fit placement (fork author's regime: experts mmap'd, GPU computes, per-use H2D)",
            "FATE-on runs force FATE_NO_STAGING=1: the shipped staging buffer races its own async H2D and corrupts output from token 1",
            "mmap (no --no-mmap): --no-mmap + partial cudaHostRegister pins 49/144 tensors, enabling a true-async H2D lifetime crash on this 8 GB placement",
            "in-repo reference: stock b10355 -ngl 48 --n-cpu-moe 33 = 33-35 tok/s (docs/moe-track-plan.md)",
            "in-repo LRU sim from real traces: ~98% hit at 0.68 GB (results/moe-locality/cache-cost-model.json)",
            "parity-gate snapshot (95-token evals): off 27.67 t/s, on(no-staging) 3.98 t/s at 56.46% hit; stock b10355 26.46 t/s",
            "prefill regression is a known fork limitation (author reports 4 t/s)",
        ],
        "records": records,
    }
    out = OUT_DIR / "fate-repro.json"
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")
    summarize(records)


if __name__ == "__main__":
    main()
