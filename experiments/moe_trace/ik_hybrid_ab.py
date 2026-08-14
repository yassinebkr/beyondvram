"""Next-experiments #4 follow-up: engine x model A/B for the IQ2 hybrid.

The mainline speed gate measured the mid24-IQ2_XXS hybrid at parity (0.988x)
despite a 1.81x roofline byte advantage at its capacity optimum: mainline's
IQ2_XXS single-token CPU path is compute-bound (~14-15 GB/s effective vs the
26.6 GB/s memory constant — sweep delivers 0.58-0.70x roofline where the
Q4_K_M baseline delivers ~1.1x). ik_llama.cpp's iqk CPU kernels are the
fork's claim to fame; this A/B tests whether the byte cut converts to speed
under ik, on the same machine, same session, drift-canceling interleave.

Arms (3 rounds, alternating order): mainline/ik x original/hybrid, each at
its measured best placement (original: -ngl 48 --n-cpu-moe 33; hybrid: 28).
Protocol: -p 128 -n 32 -r 3, mmap. Failures recorded as status rows.

Child stdout/stderr stream live by default (--quiet suppresses). Ctrl+C
terminates the running child, writes partial results to the output JSON with
status="interrupted", and exits with code 130.

Output: results/moe-locality/ik-hybrid-ab.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "system"))
from common import run_live, ts  # noqa: E402

BENCH = {
    "mainline": ROOT / "tools/llama.cpp-b10355/llama-bench.exe",
    "ik": ROOT / "tools/ik_llama.cpp/build/bin/llama-bench.exe",
}
ORIG = ROOT / "models/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf"
HYBRID = ROOT / "models/Qwen3-30B-A3B-GGUF/qwen3-30b-a3b-mid24-iq2xxs.gguf"
OUT = ROOT / "results/moe-locality/ik-hybrid-ab.json"

ARMS = [
    ("mainline-original", "mainline", ORIG, 33),
    ("mainline-hybrid", "mainline", HYBRID, 28),
    ("ik-original", "ik", ORIG, 33),
    ("ik-hybrid", "ik", HYBRID, 28),
]
ROUNDS = 3


def run_arm(engine: str, model: Path, k: int, label: str, quiet: bool) -> dict:
    cmd = [str(BENCH[engine]), "-m", str(model), "-ngl", "48", "--n-cpu-moe", str(k),
           "-p", "128", "-n", "32", "-r", "3", "-o", "json"]
    res = run_live(cmd, label, quiet=quiet, timeout=1800)
    rec = {"engine": engine, "model": model.name, "n_cpu_moe": k,
           "rc": res["rc"], "wall_s": res["wall_s"]}
    if res["rc"] != 0:
        rec["status"] = "error"
        rec["stderr_tail"] = res["stderr"][-300:]
        print(f"[{ts()}] {label}: FAILED rc={res['rc']}", flush=True)
        return rec
    try:
        for row in json.loads(res["stdout"]):
            if row.get("n_gen", 0) > 0:
                rec["tg"] = round(row["avg_ts"], 2)
            elif row.get("n_prompt", 0) > 0:
                rec["pp"] = round(row["avg_ts"], 2)
        rec["status"] = "ok"
    except json.JSONDecodeError:
        rec["status"] = "parse_error"
    print(f"[{ts()}] {label}: tg={rec.get('tg')} pp={rec.get('pp')}",
          flush=True)
    return rec


def write_payload(records: list[dict], status: str) -> dict:
    summary = {}
    for name, *_ in ARMS:
        vals = [x["tg"] for x in records if x["arm"] == name and x.get("tg")]
        if vals:
            summary[name] = {"tg_mean": round(sum(vals) / len(vals), 2), "n": len(vals)}
    payload = {
        "status": status,
        "tool": "llama-bench; mainline b10355 CI vs ik 981e5ea local MSVC+CUDA",
        "protocol": "-ngl 48, -p 128 -n 32 -r 3, arms alternating per round",
        "hypothesis": "mainline IQ2_XXS CPU gemv is compute-bound; ik iqk kernels "
                      "may convert the hybrid's byte cut into tg",
        "summary_tg": summary,
        "records": records,
    }
    OUT.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quiet", action="store_true",
                    help="suppress live child output (banners still print)")
    args = ap.parse_args()

    n_total = ROUNDS * len(ARMS)
    records: list[dict] = []
    print(f"[{ts()}] engine x model A/B: {len(ARMS)} arms x {ROUNDS} rounds = "
          f"{n_total} runs, alternating order per round", flush=True)
    print(f"[{ts()}] protocol: -ngl 48 -p 128 -n 32 -r 3, mmap; arms: "
          + ", ".join(f"{name} (k={k})" for name, _, _, k in ARMS), flush=True)
    print(f"[{ts()}] output -> {OUT} (Ctrl+C writes partial results with "
          f"status=\"interrupted\")", flush=True)
    try:
        step = 0
        for r in range(ROUNDS):
            order = ARMS if r % 2 == 0 else list(reversed(ARMS))
            for name, engine, model, k in order:
                step += 1
                label = f"[step {step}/{n_total}] round {r} {name}"
                print(f"[{ts()}] {label}: measuring", flush=True)
                records.append({"round": r, "arm": name,
                                **run_arm(engine, model, k, label, args.quiet)})
    except KeyboardInterrupt:
        write_payload(records, status="interrupted")
        print(f"\n[{ts()}] interrupted — partial results ({len(records)} "
              f"rows) -> {OUT}", flush=True)
        sys.exit(130)

    payload = write_payload(records, status="complete")
    print(f"\n[{ts()}] wrote {OUT}", flush=True)
    for name, s in payload["summary_tg"].items():
        print(f"  {name}: {s['tg_mean']} tok/s", flush=True)


if __name__ == "__main__":
    main()
