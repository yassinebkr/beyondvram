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

Ctrl+C terminates the running server, writes partial results to the output
JSON with status="interrupted", and exits with code 130.

Output: results/gpt-oss/speculative-bench.json + spec-server-*.log

N-gram mode (`--ngram`): no draft model — the server's built-in
`--spec-type ngram-simple|ngram-mod` self-speculation on a repetitive-content
prompt (code-shaped output), baseline `no_spec`, same placement and protocol.
Writes speculative-bench-ngram.json. b10355 llama-server --spec-type support:
ngram-simple, ngram-map-k, ngram-map-k4v, ngram-mod, ngram-cache.
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

PROMPT_REPETITIVE = (
    "Write a Python module defining twenty dataclasses, one per common farm "
    "animal. Every dataclass has the same fields (name: str, legs: int, "
    "habitat: str, diet: str) and an identical describe method that returns "
    "a one-line summary. Follow exactly the same pattern for all twenty."
)

NGRAM_CONFIGS = [
    ("no_spec", []),
    ("ngram_simple", ["--spec-type", "ngram-simple"]),
    ("ngram_mod", ["--spec-type", "ngram-mod"]),
]


def start_server(extra: list[str], log_path: Path) -> subprocess.Popen | None:
    log = open(log_path, "w", encoding="utf-8", errors="replace")
    cmd = [str(SERVER), "-m", str(MODEL), "-ngl", "24", "--n-cpu-moe", "10",
           "--port", str(PORT), "--no-webui", *extra]
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
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
                    return None
                time.sleep(2)
        terminate_child(proc)
        return None
    except KeyboardInterrupt:
        terminate_child(proc)
        raise


def completion(prompt: str) -> dict:
    body = json.dumps({
        "prompt": prompt, "n_predict": N_PREDICT,
        "temperature": 0.0, "seed": 42, "cache_prompt": False,
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def build_payload(records: list[dict], status: str, spec_mode: str,
                  prompt: str, reps: int,
                  configs: list[tuple[str, list[str]]],
                  interleave: bool) -> dict:
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
    return {
        "status": status,
        "server": "llama-server b10355 (unmodified)",
        "spec_mode": spec_mode,
        "model": MODEL.name,
        "draft": DRAFT.name if spec_mode == "eagle3" else None,
        "placement": "-ngl 24 --n-cpu-moe 10 (measured optimum)",
        "prompt": prompt, "n_predict": N_PREDICT, "reps": reps,
        "protocol": ("rep 0 is warmup (kept, excluded from summary); greedy, seed 42"
                     + ("; arms interleaved rep-outer, alternating order (drift-canceling)"
                        if interleave else
                        "; config-outer batching (pre-drift-discovery protocol)")),
        "expectation": ("weak-to-negative per Track-1 (29.0 vs 29.5) and the survey (-19% to +2% on CPU-expert rigs)"
                        if spec_mode == "eagle3" else
                        "repetitive content favors lookup speculation; decision rule in docs/next-experiments.md experiment 5"),
        "records": records,
        "summary_warm": summary,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reps = REPS
    configs = CONFIGS
    out = OUT
    prompt = PROMPT
    spec_mode = "eagle3"
    if "--ngram" in sys.argv:
        configs = NGRAM_CONFIGS
        prompt = PROMPT_REPETITIVE
        spec_mode = "ngram"
        out = OUT_DIR / "speculative-bench-ngram.json"
    if "--reps" in sys.argv:
        reps = int(sys.argv[sys.argv.index("--reps") + 1])
    if "--configs" in sys.argv:
        wanted = set(sys.argv[sys.argv.index("--configs") + 1].split(","))
        configs = [c for c in CONFIGS if c[0] in wanted]
    if "--tag" in sys.argv:
        tag = sys.argv[sys.argv.index("--tag") + 1]
        out = OUT_DIR / f"speculative-bench-{tag}.json"
    interleave = "--interleave" in sys.argv
    n_runs = reps * len(configs) if interleave else len(configs)
    records: list[dict] = []
    print(f"[{ts()}] speculative bench: mode={spec_mode}, {len(configs)} "
          f"configs, {reps} reps/config, n_predict={N_PREDICT}, port {PORT}",
          flush=True)
    print(f"[{ts()}] protocol: {'rep-outer interleave, alternating order' if interleave else 'config-outer batching'} "
          f"({n_runs} server runs)", flush=True)
    print(f"[{ts()}] server logs -> {OUT_DIR / 'spec-server-*.log'} "
          f"(tail -f to watch live)", flush=True)
    print(f"[{ts()}] output -> {out} (Ctrl+C writes partial results)",
          flush=True)
    try:
        if interleave:
            # Rep-outer / config-inner with alternating order, one server lifetime
            # per arm: pure-CPU llama-bench showed ~15% session-level drift on this
            # machine (docs/next-experiments.md), so config-outer batching aliases
            # drift into the config effect. Interleaving cancels it.
            run = 0
            for rep in range(reps):
                order = configs if rep % 2 == 0 else list(reversed(configs))
                for name, extra in order:
                    run += 1
                    label = f"[run {run}/{n_runs}] {name} rep{rep}"
                    log_path = OUT_DIR / f"spec-server-{name}-rep{rep}.log"
                    print(f"[{ts()}] {label}: starting server "
                          f"(log: {log_path})", flush=True)
                    proc = start_server(extra, log_path)
                    if proc is None:
                        records.append({"config": name, "rep": rep, "status": "server_died",
                                        "log": log_path.name})
                        print(f"[{ts()}] {label}: server died at startup; "
                              f"see {log_path.name}", flush=True)
                        continue
                    try:
                        result = completion(prompt)
                        timings = result.get("timings", {})
                        records.append({"config": name, "rep": rep, "status": "ok",
                                        "predicted_per_second": timings.get("predicted_per_second"),
                                        "prompt_per_second": timings.get("prompt_per_second"),
                                        "timings": timings})
                        print(f"[{ts()}] {label}: "
                              f"gen={timings.get('predicted_per_second')} tok/s",
                              flush=True)
                    finally:
                        terminate_child(proc)
        else:
            for i, (name, extra) in enumerate(configs, 1):
                label = f"[config {i}/{len(configs)}] {name}"
                log_path = OUT_DIR / f"spec-server-{name}.log"
                print(f"[{ts()}] {label}: starting server (log: {log_path})",
                      flush=True)
                proc = start_server(extra, log_path)
                if proc is None:
                    records.append({"config": name, "status": "server_died",
                                    "log": log_path.name})
                    print(f"[{ts()}] {label}: server died at startup; "
                          f"see {log_path.name}", flush=True)
                    continue
                try:
                    for rep in range(reps):
                        result = completion(prompt)
                        timings = result.get("timings", {})
                        record = {"config": name, "rep": rep, "status": "ok",
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
        payload = build_payload(records, "interrupted", spec_mode, prompt,
                                reps, configs, interleave)
        out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        print(f"\n[{ts()}] interrupted — partial results ({len(records)} "
              f"rows) -> {out}", flush=True)
        sys.exit(130)

    payload = build_payload(records, "complete", spec_mode, prompt, reps,
                            configs, interleave)
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"\n[{ts()}] wrote {out}", flush=True)
    for name, s in payload["summary_warm"].items():
        print(f"  {name}: {s['mean']} +/- {s['stdev']} tok/s (n={s['n']})",
              flush=True)


if __name__ == "__main__":
    main()
