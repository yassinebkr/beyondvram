"""Speculative decoding measurement via llama-server (b10355).

llama-cli at b10355 parses -md but never uses it (no speculative code in
tools/cli/); speculative decoding lives in llama-server. This script starts
the server with and without a draft model, POSTs fixed greedy completion
requests, and compares timings.tokens_per_second.

Server stdout/stderr are discarded (DEVNULL, no per-config log file). Ctrl+C
terminates the running server, writes partial results to the output JSON
with status="interrupted", and exits with code 130.

Output: results/moe-locality/speculative-bench.json
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "system"))
from common import terminate_child, ts  # noqa: E402

SERVER = ROOT / "tools/llama.cpp-b10355/llama-server.exe"
MODEL = ROOT / "models/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf"
DRAFT = ROOT / "models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf"
OUT = ROOT / "results/moe-locality/speculative-bench.json"
PORT = 8099

PROMPT = ("Write a short story about a lighthouse keeper who finds a strange "
          "letter. The story should have a beginning, middle, and end.")
N_PREDICT = 128
REPS = 4

CONFIGS = [
    ("no_draft", []),
    ("draft_0.6b_q8", ["-md", str(DRAFT), "-ngld", "99",
                       "--spec-draft-n-max", "8"]),
]


def start_server(extra: list[str]) -> subprocess.Popen:
    cmd = [str(SERVER), "-m", str(MODEL), "-ngl", "48", "--n-cpu-moe", "33",
           "--port", str(PORT), "--no-webui", *extra]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            stdin=subprocess.DEVNULL)
    deadline = time.time() + 600
    try:
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                    if r.status == 200:
                        return proc
            except Exception:
                if proc.poll() is not None:
                    raise SystemExit(f"server died, rc={proc.returncode}")
                time.sleep(2)
        raise SystemExit("server did not become healthy in 600s")
    except KeyboardInterrupt:
        terminate_child(proc)
        raise


def completion() -> dict:
    body = json.dumps({
        "prompt": PROMPT, "n_predict": N_PREDICT,
        "temperature": 0.0, "seed": 42, "cache_prompt": False,
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def write_payload(records: list[dict], status: str) -> None:
    OUT.write_text(json.dumps({"status": status, "prompt": PROMPT,
                               "n_predict": N_PREDICT, "records": records},
                              indent=1), encoding="utf-8")


def main() -> None:
    records: list[dict] = []
    print(f"[{ts()}] speculative bench: {len(CONFIGS)} configs "
          f"({', '.join(name for name, _ in CONFIGS)}), {REPS} reps/config, "
          f"n_predict={N_PREDICT}, port {PORT}", flush=True)
    print(f"[{ts()}] server stdout/stderr: discarded (DEVNULL, no log file)",
          flush=True)
    print(f"[{ts()}] output -> {OUT} (Ctrl+C writes partial results)",
          flush=True)
    try:
        for i, (name, extra) in enumerate(CONFIGS, 1):
            label = f"[config {i}/{len(CONFIGS)}] {name}"
            print(f"[{ts()}] {label}: starting server", flush=True)
            proc = start_server(extra)
            try:
                for rep in range(REPS):
                    result = completion()
                    timings = result.get("timings", {})
                    record = {"config": name, "rep": rep,
                              "predicted_per_second": timings.get("predicted_per_second"),
                              "prompt_per_second": timings.get("prompt_per_second"),
                              "timings": timings}
                    records.append(record)
                    print(f"[{ts()}] {label} rep{rep}: "
                          f"gen={timings.get('predicted_per_second')} tok/s",
                          flush=True)
            finally:
                terminate_child(proc)
    except KeyboardInterrupt:
        write_payload(records, status="interrupted")
        print(f"\n[{ts()}] interrupted — partial results ({len(records)} "
              f"rows) -> {OUT}", flush=True)
        sys.exit(130)
    write_payload(records, status="complete")
    print(f"\n[{ts()}] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
