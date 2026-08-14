"""Experiment 3 (docs/next-experiments.md): ik_llama.cpp vs mainline b10355 A/B.

Question: does the performance-focused fork beat mainline b10355 on this
CPU-heavy MoE regime (gpt-oss-20b MXFP4, RTX 3070 Ti 8 GiB + DDR4)? Both
engines run the identical llama-bench protocol at placements (0,0), (24,24),
and (24,10), arms alternating per round (drift-canceling, see "Cross-cutting:
session drift" in docs/next-experiments.md). Model page cache is re-warmed
before every arm. ik is a secondary setup: the pinned b10355 tree and the
unmodified CI binaries stay untouched either way.

Binaries: mainline = tools/llama.cpp-b10355/llama-bench.exe (official CI,
MSVC); ik = tools/ik_llama.cpp/build/bin/llama-bench.exe (local MSVC + CUDA
13.3, sm_86, HEAD pinned in the results JSON). Decision rule: a sustained
>=10% generation advantage at equal placement earns a full grid; otherwise
recorded as parity/negative and closed.

Per-run child stdout/stderr go to log files (raw evidence, not streamed);
each arm banner prints the log paths so they can be tailed live. Ctrl+C
terminates the running child, writes the manifest with
status="interrupted", and exits with code 130.

Output: results/gpt-oss/ik-ab-{engine}-{placement}-r{round}.json(+stderr).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "system"))
from common import terminate_child, ts  # noqa: E402

MODEL = ROOT / "models/gpt-oss-20b-GGUF/gpt-oss-20b-MXFP4.gguf"
OUT_DIR = ROOT / "results/gpt-oss"

ENGINES = {
    "mainline": ROOT / "tools/llama.cpp-b10355/llama-bench.exe",
    "ik": ROOT / "tools/ik_llama.cpp/build/bin/llama-bench.exe",
}

# (ngl, n_cpu_moe) placements: pure CPU, all experts CPU, measured optimum.
PLACEMENTS = {
    "cpu00": (0, 0),
    "moe24": (24, 24),
    "moe10": (24, 10),
}

MANIFEST = OUT_DIR / "ik-ab-manifest.json"


def warm_cache(path: Path) -> None:
    """Read the model fully so every arm sees a warm page cache."""
    with open(path, "rb") as f:
        while f.read(1 << 24):
            pass


def run_arm(engine: str, placement: str, round_idx: int,
            step: int, total: int) -> Path:
    ngl, k = PLACEMENTS[placement]
    out = OUT_DIR / f"ik-ab-{engine}-{placement}-r{round_idx}.json"
    err = out.with_suffix(".stderr.txt")
    label = f"[step {step}/{total}] {engine} {placement} r{round_idx}"
    print(f"[{ts()}] {label}: warming cache", flush=True)
    warm_cache(MODEL)
    cmd = [str(ENGINES[engine]), "-m", str(MODEL),
           "-ngl", str(ngl), "--n-cpu-moe", str(k),
           "-p", "128", "-n", "32", "-r", "3", "-o", "json"]
    print(f"[{ts()}] {label}: running (logs: {out} , {err})", flush=True)
    with open(out, "w") as fo, open(err, "w") as fe:
        proc = subprocess.Popen(cmd, stdout=fo, stderr=fe)
        try:
            rc = proc.wait()
        except KeyboardInterrupt:
            print(f"\n[{ts()}] {label} interrupt — terminating child",
                  flush=True)
            terminate_child(proc)
            raise
    if rc != 0:
        print(f"[{ts()}] {label} FAILED rc={rc}", flush=True)
    else:
        print(f"[{ts()}] {label} done", flush=True)
    return out


def ik_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT / "tools/ik_llama.cpp"), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def write_manifest(placements: list[str], rounds: int, status: str) -> None:
    manifest = {
        "status": status,
        "experiment": "next-experiments.md #3: ik_llama.cpp vs mainline b10355",
        "model": MODEL.name,
        "protocol": "-p 128 -n 32 -r 3, arms alternating per round, cache re-warmed per arm",
        "engines": {k: str(v) for k, v in ENGINES.items()},
        "ik_commit": ik_commit(),
        "rounds": rounds, "placements": placements,
        "written": time.strftime("%Y-%m-%d %H:%M %z"),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--placements", default="cpu00,moe24,moe10",
                    help="comma list from: " + ",".join(PLACEMENTS))
    args = ap.parse_args()

    for name, exe in ENGINES.items():
        if not exe.exists():
            ap.error(f"missing {name} binary: {exe}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    placements = [p.strip() for p in args.placements.split(",") if p.strip()]
    unknown = [p for p in placements if p not in PLACEMENTS]
    if unknown:
        ap.error(f"unknown placements: {unknown}")

    total = args.rounds * len(placements) * len(ENGINES)
    print(f"[{ts()}] ik A/B: {args.rounds} rounds x {len(placements)} "
          f"placements x {len(ENGINES)} engines = {total} runs, "
          f"model {MODEL.name}", flush=True)
    print(f"[{ts()}] per-run logs -> {OUT_DIR}/ik-ab-<engine>-<placement>"
          f"-r<round>.json(+.stderr.txt)", flush=True)
    print(f"[{ts()}] manifest -> {MANIFEST} (Ctrl+C writes partial results)",
          flush=True)

    order = list(ENGINES)
    step = 0
    try:
        for r in range(args.rounds):
            engines = order if r % 2 == 0 else list(reversed(order))
            for placement in placements:
                print(f"[{ts()}] [ik-ab] round {r} {placement}", flush=True)
                for engine in engines:
                    step += 1
                    run_arm(engine, placement, r, step, total)
    except KeyboardInterrupt:
        write_manifest(placements, args.rounds, status="interrupted")
        print(f"\n[{ts()}] interrupted after {step}/{total} runs — partial "
              f"manifest -> {MANIFEST}", flush=True)
        print(f"[{ts()}] completed per-run logs are in {OUT_DIR}", flush=True)
        sys.exit(130)

    write_manifest(placements, args.rounds, status="complete")
    print(f"[{ts()}] [ik-ab] all done — manifest -> {MANIFEST}", flush=True)


if __name__ == "__main__":
    main()
