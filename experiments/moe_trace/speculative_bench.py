"""Speculative decoding measurement via llama-server (b10355).

llama-cli at b10355 parses -md but never uses it (no speculative code in
tools/cli/); speculative decoding lives in llama-server. This script starts
the server with and without a draft model, POSTs fixed greedy completion
requests, and compares timings.tokens_per_second.

Output: results/moe-locality/speculative-bench.json
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SERVER = ROOT / "tools/llama.cpp-b10355/llama-server.exe"
MODEL = ROOT / "models/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf"
DRAFT = ROOT / "models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf"
OUT = ROOT / "results/moe-locality/speculative-bench.json"
PORT = 8099

PROMPT = ("Write a short story about a lighthouse keeper who finds a strange "
          "letter. The story should have a beginning, middle, and end.")
N_PREDICT = 128
REPS = 4


def start_server(extra: list[str]) -> subprocess.Popen:
    cmd = [str(SERVER), "-m", str(MODEL), "-ngl", "48", "--n-cpu-moe", "33",
           "--port", str(PORT), "--no-webui", *extra]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            stdin=subprocess.DEVNULL)
    deadline = time.time() + 600
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


def completion() -> dict:
    body = json.dumps({
        "prompt": PROMPT, "n_predict": N_PREDICT,
        "temperature": 0.0, "seed": 42, "cache_prompt": False,
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def main() -> None:
    records = []
    for name, extra in [("no_draft", []),
                        ("draft_0.6b_q8", ["-md", str(DRAFT), "-ngld", "99",
                                           "--spec-draft-n-max", "8"])]:
        print(f"[server] {name}", flush=True)
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
                print(f"  rep{rep}: gen={timings.get('predicted_per_second')} tok/s", flush=True)
        finally:
            proc.terminate()
            proc.wait(timeout=30)
    OUT.write_text(json.dumps({"prompt": PROMPT, "n_predict": N_PREDICT,
                               "records": records}, indent=1), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
