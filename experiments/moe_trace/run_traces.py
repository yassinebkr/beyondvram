"""Run the pinned llama-moe-trace build over a fixed prompt set.

Each run writes one JSONL trace (ffn_moe_topk records) plus the generated
text. Greedy sampling (temp 0) and a fixed seed keep runs reproducible.

Child stdout/stderr stream live by default (--quiet suppresses). Ctrl+C
terminates the running child and exits with code 130; prompts completed
before the interrupt keep their already-written trace/gen files.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "system"))
from common import run_live, ts  # noqa: E402

DEFAULT_EXE = ROOT / "tools" / "llama.cpp-source" / "build" / "bin" / "llama-moe-trace.exe"
DEFAULT_MODEL = ROOT / "models" / "Qwen3-30B-A3B-GGUF" / "Qwen3-30B-A3B-Q4_K_M.gguf"
OUT_DIR = ROOT / "results" / "moe-locality"

# Fixed, deliberately diverse prompt set (prose, code, math, multilingual, chat).
PROMPTS = [
    ("prose-intro", "Write a short introduction to the history of the printing press."),
    ("code-python", "Write a Python function that returns the n-th Fibonacci number and explain it."),
    ("math-proof", "Explain step by step why the square root of 2 is irrational."),
    ("chat-advice", "What are three practical ways to reduce home electricity usage in winter?"),
    ("code-cpp", "Write a C++ program that sorts a vector of integers using merge sort."),
    ("prose-story", "Tell a short story about a lighthouse keeper who finds a strange letter."),
    ("french", "Explique en quelques phrases pourquoi le ciel est bleu."),
    ("tech-explain", "Explain how NVMe solid state drives differ from SATA SSDs at the protocol level."),
    ("reasoning", "A train travels 120 km in 1.5 hours, then 80 km in 0.5 hours. What is its average speed for the whole journey? Show your reasoning."),
    ("chat-recipe", "Give a simple recipe for vegetable soup with exact ingredient amounts."),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, default=DEFAULT_EXE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--n-predict", type=int, default=128)
    parser.add_argument("--gpu-layers", type=int, default=18)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only", nargs="*", default=None,
                        help="restrict to these prompt ids")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR,
                        help="trace/gen output directory (default: Track-1 results dir)")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress live child output (banners still print)")
    args = parser.parse_args()

    if not args.exe.exists():
        raise SystemExit(f"trace executable not found: {args.exe}")
    if not args.model.exists():
        raise SystemExit(f"model not found: {args.model}")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    total = sum(1 for pid, _ in PROMPTS if not args.only or pid in args.only)
    print(f"[{ts()}] trace capture: {total} prompts x {args.n_predict} tokens, "
          f"-ngl {args.gpu_layers}, seed {args.seed}, temp 0", flush=True)
    print(f"[{ts()}] output -> {out_dir} (Ctrl+C terminates the running child; "
          f"completed prompts keep their files)", flush=True)

    step = 0
    done = 0
    try:
        for prompt_id, prompt in PROMPTS:
            if args.only and prompt_id not in args.only:
                continue
            step += 1
            trace_path = out_dir / f"trace-{prompt_id}.jsonl"
            gen_path = out_dir / f"gen-{prompt_id}.txt"
            # run_live inherits the process environment; set MOE_TRACE_OUT so
            # the child sees the same env the old env=dict(os.environ)+override
            # construction produced.
            os.environ["MOE_TRACE_OUT"] = str(trace_path)
            command = [
                str(args.exe),
                "-m", str(args.model),
                "-p", prompt,
                "-n", str(args.n_predict),
                "-ngl", str(args.gpu_layers),
                "--seed", str(args.seed),
                "--temp", "0",
            ]
            label = f"[step {step}/{total}] {prompt_id}"
            print(f"[{ts()}] {label} tracing -> {trace_path.name}", flush=True)
            res = run_live(command, label, quiet=args.quiet)
            gen_path.write_text(res["stdout"], encoding="utf-8")
            (out_dir / f"gen-{prompt_id}.stderr.txt").write_text(res["stderr"], encoding="utf-8")
            records = sum(1 for _ in trace_path.open(encoding="utf-8")) if trace_path.exists() else 0
            print(f"[{ts()}] {label} rc={res['rc']} records={records} "
                  f"wall={res['wall_s']}s", flush=True)
            if res["rc"] != 0:
                sys.exit(f"trace run failed for {prompt_id}; see {gen_path.stem}.stderr.txt")
            done += 1
    except KeyboardInterrupt:
        print(f"\n[{ts()}] interrupted after {done}/{total} prompts — "
              f"completed traces remain in {out_dir}", flush=True)
        sys.exit(130)

    print(f"[{ts()}] done: {done}/{total} prompts traced -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
