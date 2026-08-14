"""Parity check: llama-moe-trace (pinned dd1ea52 + trace hook) vs unmodified b10355 llama-completion.

Runs both sequentially on the same fixed prompt with greedy decoding and compares the
32-token continuation. Writes raw outputs to results/moe-locality/parity-*.txt.
(llama-cli is unusable for this at b10355: it stays in an interactive input loop even
with -no-cnv and stdin closed; llama-completion is the non-interactive equivalent.)

Child stdout/stderr stream live by default (--quiet suppresses). Ctrl+C
terminates the running child and exits with code 130.
"""
import argparse
import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "benchmarks" / "system"))
from common import run_live, ts  # noqa: E402

prompt = "The capital of France is"
common = [
    "-m", str(root / "models/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf"),
    "-p", prompt, "-n", "32", "-ngl", "18", "--seed", "42", "--temp", "0",
]

out_dir = root / "results/moe-locality"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quiet", action="store_true",
                    help="suppress live child output (banners still print)")
    args = ap.parse_args()

    out_dir.mkdir(parents=True, exist_ok=True)
    # run_live inherits the process environment; the instrumented build reads
    # MOE_TRACE_OUT, so set it process-wide exactly as the old env copy did.
    os.environ["MOE_TRACE_OUT"] = str(out_dir / "parity-trace.jsonl")

    print(f"[{ts()}] parity check: 2 runs, prompt {prompt!r}, 32 tokens, "
          f"greedy, seed 42, -ngl 18", flush=True)
    print(f"[{ts()}] output -> {out_dir}{os.sep}parity-*.txt "
          f"(Ctrl+C terminates the running child)", flush=True)

    try:
        label = "[step 1/2] llama-moe-trace (instrumented)"
        print(f"[{ts()}] {label}", flush=True)
        trace = run_live(
            [str(root / "tools/llama.cpp-source/build/bin/llama-moe-trace.exe"), *common],
            label, quiet=args.quiet, timeout=1800)
        (out_dir / "parity-trace-out.txt").write_text(trace["stdout"], encoding="utf-8")
        (out_dir / "parity-trace-stderr.txt").write_text(trace["stderr"], encoding="utf-8")
        if trace["rc"] != 0:
            sys.exit("trace run failed; see parity-trace-stderr.txt")

        label = "[step 2/2] llama-completion b10355 (reference)"
        print(f"[{ts()}] {label}", flush=True)
        cli = run_live(
            [str(root / "tools/llama.cpp-b10355/llama-completion.exe"), *common, "-no-cnv"],
            label, quiet=args.quiet, timeout=1800)
        (out_dir / "parity-cli-out.txt").write_text(cli["stdout"], encoding="utf-8")
        (out_dir / "parity-cli-stderr.txt").write_text(cli["stderr"], encoding="utf-8")
    except KeyboardInterrupt:
        print(f"\n[{ts()}] interrupted — completed parity outputs remain in "
              f"{out_dir}", flush=True)
        sys.exit(130)

    t_out, c_out = trace["stdout"].strip(), cli["stdout"].strip()
    print("TRACE OUT:", repr(t_out[:300]), flush=True)
    print("CLI   OUT:", repr(c_out[:300]), flush=True)
    match = t_out == c_out or t_out in c_out or c_out in t_out
    print("MATCH:", match, flush=True)
    sys.exit(0 if match else 1)


if __name__ == "__main__":
    main()
