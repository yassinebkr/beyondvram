"""Parity check: llama-moe-trace (pinned dd1ea52 + trace hook) vs unmodified b10355 llama-completion.

Runs both sequentially on the same fixed prompt with greedy decoding and compares the
32-token continuation. Writes raw outputs to results/moe-locality/parity-*.txt.
(llama-cli is unusable for this at b10355: it stays in an interactive input loop even
with -no-cnv and stdin closed; llama-completion is the non-interactive equivalent.)
"""
import os
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
env = dict(os.environ)
env["MOE_TRACE_OUT"] = str(root / "results/moe-locality/parity-trace.jsonl")

prompt = "The capital of France is"
common = [
    "-m", str(root / "models/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf"),
    "-p", prompt, "-n", "32", "-ngl", "18", "--seed", "42", "--temp", "0",
]

out_dir = root / "results/moe-locality"
out_dir.mkdir(parents=True, exist_ok=True)

trace = subprocess.run(
    [str(root / "tools/llama.cpp-source/build/bin/llama-moe-trace.exe"), *common],
    env=env, capture_output=True, text=True, timeout=1800, stdin=subprocess.DEVNULL,
)
print("trace rc:", trace.returncode, flush=True)
(out_dir / "parity-trace-out.txt").write_text(trace.stdout, encoding="utf-8")
(out_dir / "parity-trace-stderr.txt").write_text(trace.stderr, encoding="utf-8")
if trace.returncode != 0:
    sys.exit("trace run failed; see parity-trace-stderr.txt")

cli = subprocess.run(
    [str(root / "tools/llama.cpp-b10355/llama-completion.exe"), *common, "-no-cnv"],
    env=env, capture_output=True, text=True, timeout=1800, stdin=subprocess.DEVNULL,
)
print("cli rc:", cli.returncode, flush=True)
(out_dir / "parity-cli-out.txt").write_text(cli.stdout, encoding="utf-8")
(out_dir / "parity-cli-stderr.txt").write_text(cli.stderr, encoding="utf-8")

t_out, c_out = trace.stdout.strip(), cli.stdout.strip()
print("TRACE OUT:", repr(t_out[:300]))
print("CLI   OUT:", repr(c_out[:300]))
match = t_out == c_out or t_out in c_out or c_out in t_out
print("MATCH:", match)
sys.exit(0 if match else 1)
