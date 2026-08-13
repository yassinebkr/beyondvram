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


def warm_cache(path: Path) -> None:
    """Read the model fully so every arm sees a warm page cache."""
    with open(path, "rb") as f:
        while f.read(1 << 24):
            pass


def run_arm(engine: str, placement: str, round_idx: int) -> Path:
    ngl, k = PLACEMENTS[placement]
    out = OUT_DIR / f"ik-ab-{engine}-{placement}-r{round_idx}.json"
    err = out.with_suffix(".stderr.txt")
    warm_cache(MODEL)
    cmd = [str(ENGINES[engine]), "-m", str(MODEL),
           "-ngl", str(ngl), "--n-cpu-moe", str(k),
           "-p", "128", "-n", "32", "-r", "3", "-o", "json"]
    with open(out, "w") as fo, open(err, "w") as fe:
        proc = subprocess.run(cmd, stdout=fo, stderr=fe)
    if proc.returncode != 0:
        print(f"[ik-ab] {engine} {placement} r{round_idx} FAILED rc={proc.returncode}",
              file=sys.stderr)
    else:
        print(f"[ik-ab] {engine} {placement} r{round_idx} done", file=sys.stderr)
    return out


def ik_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT / "tools/ik_llama.cpp"), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


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

    order = list(ENGINES)
    for r in range(args.rounds):
        engines = order if r % 2 == 0 else list(reversed(order))
        for placement in placements:
            print(f"[ik-ab] round {r} {placement}", file=sys.stderr)
            for engine in engines:
                run_arm(engine, placement, r)

    manifest = {
        "experiment": "next-experiments.md #3: ik_llama.cpp vs mainline b10355",
        "model": MODEL.name,
        "protocol": "-p 128 -n 32 -r 3, arms alternating per round, cache re-warmed per arm",
        "engines": {k: str(v) for k, v in ENGINES.items()},
        "ik_commit": ik_commit(),
        "rounds": args.rounds, "placements": placements,
        "written": time.strftime("%Y-%m-%d %H:%M %z"),
    }
    (OUT_DIR / "ik-ab-manifest.json").write_text(json.dumps(manifest, indent=1))
    print("[ik-ab] all done", file=sys.stderr)


if __name__ == "__main__":
    main()
