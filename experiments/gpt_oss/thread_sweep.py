"""gpt-oss-20b thread-count sweep: is there free DDR4 bandwidth in the CPU expert path?

Context (docs/track4-gpt-oss.md, corrected roofline): the all-experts-CPU
point (24,24) implies ~26.6 GB/s of effective DDR4 read throughput, and the
B07 pure-read benchmark measures a practical NumPy read ceiling of ~24-26
GB/s on this machine. If llama.cpp's CPU expert path is already at the
practical ceiling, no thread-count change should move generation tok/s by
more than run-to-run variance. If a thread count does move it materially,
the effective-bandwidth constant in the roofline model must be revised
upward — which would raise the gpt-oss-120b ceiling prediction too.

Same protocol as the placement grid: unmodified b10355 llama-bench,
128 prompt tokens, 32 generated, 3 repetitions, mmap, f16 KV.
Two placements: (24,24) all-experts-CPU and the (24,10) optimum.

Child stdout/stderr stream live by default (--quiet suppresses). Ctrl+C
terminates the running child, writes partial results to the output JSON with
status="interrupted", and exits with code 130.

Output: results/gpt-oss/thread-sweep.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "system"))
from common import run_live, ts  # noqa: E402

BENCH = ROOT / "tools" / "llama.cpp-b10355" / "llama-bench.exe"
MODEL = ROOT / "models" / "gpt-oss-20b-GGUF" / "gpt-oss-20b-MXFP4.gguf"
OUT = ROOT / "results" / "gpt-oss" / "thread-sweep.json"

PLACEMENTS = [(24, 24), (24, 10)]
THREADS = [4, 6, 8, 10, 12]
REPS = "3"


def write_payload(records: list[dict], status: str) -> None:
    payload = {
        "status": status,
        "bench": "llama-bench b10355 (unmodified)",
        "model": MODEL.name,
        "protocol": f"128 prompt tokens, 32 generated, {REPS} repetitions, mmap, f16 KV",
        "question": "does thread count move effective DDR4 read throughput "
                    "of the CPU expert path above the ~26.6 GB/s implied by (24,24)?",
        "references": {
            "b07_read_ceiling_gb_s": "numpy f64 sum: mt6 25.85, mt12 24.30 median",
            "implied_effective_gb_s": 26.6,
        },
        "placements": [{"ngl": n, "n_cpu_moe": k} for n, k in PLACEMENTS],
        "threads": THREADS,
        "records": records,
    }
    OUT.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quiet", action="store_true",
                    help="suppress live child output (banners still print)")
    args = ap.parse_args()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    total = len(PLACEMENTS) * len(THREADS)
    print(f"[{ts()}] thread sweep: {len(PLACEMENTS)} placements x "
          f"{len(THREADS)} thread counts = {total} runs, model {MODEL.name}, "
          f"{REPS} reps each", flush=True)
    print(f"[{ts()}] output -> {OUT} (Ctrl+C writes partial results)",
          flush=True)
    step = 0
    try:
        for ngl, ncpu in PLACEMENTS:
            for threads in THREADS:
                step += 1
                label = (f"[step {step}/{total}] ngl={ngl} "
                         f"n-cpu-moe={ncpu} -t {threads}")
                cmd = [
                    str(BENCH), "-m", str(MODEL),
                    "-ngl", str(ngl), "--n-cpu-moe", str(ncpu),
                    "-t", str(threads),
                    "-p", "128", "-n", "32", "-r", REPS, "-o", "json",
                ]
                print(f"[{ts()}] {label}: measuring", flush=True)
                res = run_live(cmd, label, quiet=args.quiet, timeout=1800)
                record = {"ngl": ngl, "n_cpu_moe": ncpu, "threads": threads,
                          "rc": res["rc"], "wall_s": res["wall_s"]}
                if res["rc"] == 0:
                    try:
                        rows = json.loads(res["stdout"])
                        gen = [r for r in rows if r.get("n_gen")]
                        pp = [r for r in rows if r.get("n_prompt")]
                        record["tg_avg_ts"] = gen[0]["avg_ts"] if gen else None
                        record["tg_stddev_ts"] = gen[0]["stddev_ts"] if gen else None
                        record["pp_avg_ts"] = pp[0]["avg_ts"] if pp else None
                        print(f"  ok: tg={record['tg_avg_ts']:.2f} tok/s",
                              flush=True)
                    except (json.JSONDecodeError, IndexError, KeyError):
                        record["status"] = "parse_error"
                        record["stdout_head"] = res["stdout"][:300]
                else:
                    record["status"] = "error"
                    record["stderr_tail"] = res["stderr"][-400:]
                    print(f"  FAILED rc={res['rc']}: {res['stderr'][-200:]!r}",
                          flush=True)
                records.append(record)
    except KeyboardInterrupt:
        write_payload(records, status="interrupted")
        print(f"\n[{ts()}] interrupted — partial results ({len(records)} "
              f"rows) -> {OUT}", flush=True)
        sys.exit(130)

    write_payload(records, status="complete")
    print(f"\n[{ts()}] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
