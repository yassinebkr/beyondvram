"""EAGLE-3 speculative-decoding measurement for gpt-oss-20b via llama-server.

Track-1 measured draft speculation negative on Qwen3-30B-A3B (29.0 vs 29.5
tok/s): verification passes re-read the same bandwidth-bound experts, so
speculation cannot amortize on CPU-expert rigs. The deep-research survey
records hybrid speculative decoding on 8 GB + CPU-experts as weak-to-negative
(-19% to +2%). This bench closes the question for gpt-oss-20b with the
official EAGLE-3 draft heads (ggml-org, Q8_0) at the measured placement
optimum (-ngl 24 --n-cpu-moe 10). Expectations: low; evidence over priors.

Same protocol as experiments/moe_trace/speculative_bench.py: llama-server
b10355, fixed greedy prompt, n_predict 128, 4 reps (first is warmup, kept but
excluded from the summary). Server stderr is captured per config so draft
engagement (or a silent fallback) is verifiable, unlike the Track-1 run.
A config whose server dies at startup is recorded as server_died, not dropped.

Output: results/gpt-oss/speculative-bench.json + spec-server-*.log
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SERVER = ROOT / "tools/llama.cpp-b10355/llama-server.exe"
MODEL = ROOT / "models/gpt-oss-20b-GGUF/gpt-oss-20b-MXFP4.gguf"
DRAFT = ROOT / "models/gpt-oss-20b-GGUF/eagle3-gpt-oss-20b-Q8_0.gguf"
OUT_DIR = ROOT / "results/gpt-oss"
OUT = OUT_DIR / "speculative-bench.json"
PORT = 8100

PROMPT = ("Write a short story about a lighthouse keeper who finds a strange "
          "letter. The story should have a beginning, middle, and end.")
N_PREDICT = 128
REPS = 4

CONFIGS = [
    ("no_draft", []),
    ("draft_gpu", ["-md", str(DRAFT), "-ngld", "99", "--spec-draft-n-max", "8"]),
    ("draft_cpu", ["-md", str(DRAFT), "-ngld", "0", "--spec-draft-n-max", "8"]),
    # -ngld 99 dies at draft load ("invalid vector subscript", b10355 bug in
    # the EAGLE-3 draft GPU-placement path); ngld 1 probes whether any GPU
    # offload of the draft survives.
    ("draft_gpu_ngld1", ["-md", str(DRAFT), "-ngld", "1", "--spec-draft-n-max", "8"]),
]


def start_server(extra: list[str], log_path: Path) -> subprocess.Popen | None:
    log = open(log_path, "w", encoding="utf-8", errors="replace")
    cmd = [str(SERVER), "-m", str(MODEL), "-ngl", "24", "--n-cpu-moe", "10",
           "--port", str(PORT), "--no-webui", *extra]
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL)
    deadline = time.time() + 600
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                if r.status == 200:
                    return proc
        except Exception:
            if proc.poll() is not None:
                return None
            time.sleep(2)
    proc.terminate()
    return None


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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reps = REPS
    configs = CONFIGS
    out = OUT
    if "--reps" in sys.argv:
        reps = int(sys.argv[sys.argv.index("--reps") + 1])
    if "--configs" in sys.argv:
        wanted = set(sys.argv[sys.argv.index("--configs") + 1].split(","))
        configs = [c for c in CONFIGS if c[0] in wanted]
    if "--tag" in sys.argv:
        tag = sys.argv[sys.argv.index("--tag") + 1]
        out = OUT_DIR / f"speculative-bench-{tag}.json"
    records = []
    if "--interleave" in sys.argv:
        # Rep-outer / config-inner with alternating order, one server lifetime
        # per arm: pure-CPU llama-bench showed ~15% session-level drift on this
        # machine (docs/next-experiments.md), so config-outer batching aliases
        # drift into the config effect. Interleaving cancels it.
        for rep in range(reps):
            order = configs if rep % 2 == 0 else list(reversed(configs))
            for name, extra in order:
                print(f"[server] {name} rep{rep}", flush=True)
                log_path = OUT_DIR / f"spec-server-{name}-rep{rep}.log"
                proc = start_server(extra, log_path)
                if proc is None:
                    records.append({"config": name, "rep": rep, "status": "server_died",
                                    "log": log_path.name})
                    print(f"  server died at startup; see {log_path.name}", flush=True)
                    continue
                try:
                    result = completion()
                    timings = result.get("timings", {})
                    records.append({"config": name, "rep": rep, "status": "ok",
                                    "predicted_per_second": timings.get("predicted_per_second"),
                                    "prompt_per_second": timings.get("prompt_per_second"),
                                    "timings": timings})
                    print(f"  rep{rep}: gen={timings.get('predicted_per_second')} tok/s", flush=True)
                finally:
                    proc.terminate()
                    proc.wait(timeout=30)
    else:
        for name, extra in configs:
            print(f"[server] {name}", flush=True)
            log_path = OUT_DIR / f"spec-server-{name}.log"
            proc = start_server(extra, log_path)
            if proc is None:
                records.append({"config": name, "status": "server_died",
                                "log": log_path.name})
                print(f"  server died at startup; see {log_path.name}", flush=True)
                continue
            try:
                for rep in range(reps):
                    result = completion()
                    timings = result.get("timings", {})
                    record = {"config": name, "rep": rep, "status": "ok",
                              "predicted_per_second": timings.get("predicted_per_second"),
                              "prompt_per_second": timings.get("prompt_per_second"),
                              "timings": timings}
                    records.append(record)
                    print(f"  rep{rep}: gen={timings.get('predicted_per_second')} tok/s", flush=True)
            finally:
                proc.terminate()
                proc.wait(timeout=30)

    payload = {
        "server": "llama-server b10355 (unmodified)",
        "model": MODEL.name, "draft": DRAFT.name,
        "placement": "-ngl 24 --n-cpu-moe 10 (measured optimum)",
        "prompt": PROMPT, "n_predict": N_PREDICT, "reps": reps,
        "protocol": ("rep 0 is warmup (kept, excluded from summary); greedy, seed 42"
                     + ("; arms interleaved rep-outer, alternating order (drift-canceling)"
                        if "--interleave" in sys.argv else
                        "; config-outer batching (pre-drift-discovery protocol)")),
        "expectation": "weak-to-negative per Track-1 (29.0 vs 29.5) and the survey (-19% to +2% on CPU-expert rigs)",
        "records": records,
    }
    from statistics import mean, stdev
    summary = {}
    for name, _ in configs:
        vals = [r["predicted_per_second"] for r in records
                if r.get("config") == name and r.get("status") == "ok"
                and r["rep"] > 0 and r["predicted_per_second"] is not None]
        if vals:
            summary[name] = {"mean": round(mean(vals), 2),
                             "stdev": round(stdev(vals), 2) if len(vals) > 1 else 0.0,
                             "n": len(vals)}
    payload["summary_warm"] = summary
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")
    for name, s in summary.items():
        print(f"  {name}: {s['mean']} +/- {s['stdev']} tok/s (n={s['n']})")


if __name__ == "__main__":
    main()
