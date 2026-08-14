"""Next-experiments #4 speed gate: placement sweep + interleaved A/B.

Phase 1 sweeps --n-cpu-moe at -ngl 48 on the mid24-IQ2_XXS hybrid (13.7 GB;
the smaller expert layers shift the capacity optimum). Placement identity is
byte-neutral — every expert layer reads top-8 per token regardless of routing,
so only total GPU-resident expert bytes matter, and the plain --n-cpu-moe
prefix split suffices (no -ot needed).

Phase 2 anchors the verdict against session drift (docs/next-experiments.md
"Cross-cutting: session drift" — cross-session effects <15% are invisible):
the hybrid's best config vs the original Q4_K_M at its measured optimum
(48,33), 3 rounds, alternating order, same session.

Protocol: unmodified b10355 llama-bench, 128 prompt / 32 gen / 3 reps, mmap,
f16 KV. OOM and failures recorded as status rows, never dropped.
Decision rule (docs/next-experiments.md #4): tg gain >=15% at best placement.

Child stdout/stderr stream live by default (--quiet suppresses). Ctrl+C
terminates the running child, writes partial results to the output JSON with
status="interrupted", and exits with code 130.

Output: results/moe-locality/mixed-precision-speed[-TAG].json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "system"))
from common import run_live, ts  # noqa: E402

BENCH = ROOT / "tools/llama.cpp-b10355/llama-bench.exe"
ORIG = ROOT / "models/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf"
DEFAULT_HYBRID = ROOT / "models/Qwen3-30B-A3B-GGUF/qwen3-30b-a3b-mid24-iq2xxs.gguf"

SWEEP_K = [24, 28, 30, 32, 36, 40, 44]
AB_ROUNDS = 3


def write_payload(out: Path, hybrid: Path, records: list[dict],
                  status: str) -> dict:
    ab_tg: dict[str, dict] = {}
    for name in ("hybrid", "original"):
        vals = [x["tg"] for x in records
                if x.get("phase") == "ab" and x.get("arm") == name and x.get("tg")]
        if vals:
            ab_tg[name] = {"mean": round(sum(vals) / len(vals), 2), "n": len(vals)}
    payload = {
        "status": status,
        "tool": "llama-bench b10355 (unmodified); -p 128 -n 32 -r 3, mmap, f16 KV",
        "hybrid": hybrid.name, "original": ORIG.name,
        "phase1_sweep_k": SWEEP_K,
        "phase2_ab_tg": ab_tg,
        "tg_gain_vs_original": (round(ab_tg["hybrid"]["mean"] / ab_tg["original"]["mean"], 3)
                                if len(ab_tg) == 2 else None),
        "decision_rule": ">=1.15 tg gain at best placement (docs/next-experiments.md #4)",
        "records": records,
    }
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return payload


def run_bench(model: Path, ngl: int, k: int, label: str, quiet: bool) -> dict:
    cmd = [str(BENCH), "-m", str(model), "-ngl", str(ngl), "--n-cpu-moe", str(k),
           "-p", "128", "-n", "32", "-r", "3", "-o", "json"]
    res = run_live(cmd, label, quiet=quiet)
    rec = {"model": model.name, "ngl": ngl, "n_cpu_moe": k,
           "rc": res["rc"], "wall_s": res["wall_s"]}
    if res["rc"] != 0:
        rec["status"] = "error"
        rec["stderr_tail"] = res["stderr"][-300:]
        print(f"[{ts()}] {label}: FAILED rc={res['rc']} (ngl {ngl}, "
              f"cpu-moe {k})", flush=True)
        return rec
    try:
        rows = json.loads(res["stdout"])
        for row in rows:
            if row.get("n_gen", 0) > 0:
                rec["tg"] = round(row["avg_ts"], 2)
            elif row.get("n_prompt", 0) > 0:
                rec["pp"] = round(row["avg_ts"], 2)
        rec["status"] = "ok"
    except json.JSONDecodeError:
        rec["status"] = "parse_error"
        rec["stdout_tail"] = res["stdout"][-300:]
    print(f"[{ts()}] {label}: ngl {ngl} cpu-moe {k}: tg={rec.get('tg')} "
          f"pp={rec.get('pp')}", flush=True)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hybrid", default=str(DEFAULT_HYBRID),
                    help="path to the mixed-precision variant GGUF")
    ap.add_argument("--tag", default="",
                    help="output suffix: mixed-precision-speed[-TAG].json")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress live child output (banners still print)")
    args = ap.parse_args()
    hybrid = Path(args.hybrid)
    out = (ROOT / "results/moe-locality/mixed-precision-speed"
           f"{'-' + args.tag if args.tag else ''}.json")
    if not hybrid.exists():
        ap.error(f"missing hybrid model: {hybrid}")

    n_ab = AB_ROUNDS * 2
    records: list[dict] = []
    print(f"[{ts()}] speed gate: hybrid={hybrid.name}", flush=True)
    print(f"[{ts()}] plan: phase 1 sweep {len(SWEEP_K)} placements, "
          f"phase 2 interleaved A/B {n_ab} runs", flush=True)
    print(f"[{ts()}] output -> {out} (Ctrl+C writes partial results)",
          flush=True)
    try:
        print(f"[{ts()}] === phase 1: hybrid placement sweep ===", flush=True)
        sweep = []
        for j, k in enumerate(SWEEP_K, 1):
            rec = run_bench(hybrid, 48, k,
                            label=f"[sweep {j}/{len(SWEEP_K)}] k={k}",
                            quiet=args.quiet)
            records.append({"phase": "sweep", **rec})
            if rec["status"] == "ok":
                sweep.append(rec)
        if not sweep:
            print(f"[{ts()}] all sweep configs failed — aborting",
                  file=sys.stderr, flush=True)
            write_payload(out, hybrid, records, status="failed")
            sys.exit(1)
        best = max(sweep, key=lambda r: r["tg"])
        print(f"[{ts()}] === phase 1 best: cpu-moe {best['n_cpu_moe']} "
              f"tg={best['tg']} ===", flush=True)

        print(f"[{ts()}] === phase 2: interleaved A/B vs original at "
              f"(48,33) ===", flush=True)
        arms = [("hybrid", hybrid, best["n_cpu_moe"]), ("original", ORIG, 33)]
        step = 0
        for r in range(AB_ROUNDS):
            order = arms if r % 2 == 0 else list(reversed(arms))
            for name, model, k in order:
                step += 1
                rec = run_bench(model, 48, k,
                                label=f"[ab {step}/{n_ab}] round {r} {name}",
                                quiet=args.quiet)
                records.append({"phase": "ab", "round": r, "arm": name, **rec})
    except KeyboardInterrupt:
        write_payload(out, hybrid, records, status="interrupted")
        print(f"\n[{ts()}] interrupted — partial results ({len(records)} "
              f"rows) -> {out}", flush=True)
        sys.exit(130)

    payload = write_payload(out, hybrid, records, status="complete")
    print(f"\n[{ts()}] wrote {out}", flush=True)
    ab_tg = payload["phase2_ab_tg"]
    if len(ab_tg) == 2:
        print(f"[{ts()}] A/B tg: hybrid {ab_tg['hybrid']['mean']} vs original "
              f"{ab_tg['original']['mean']} = "
              f"{payload['tg_gain_vs_original']}x", flush=True)


if __name__ == "__main__":
    main()
