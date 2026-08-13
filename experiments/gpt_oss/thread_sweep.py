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

Output: results/gpt-oss/thread-sweep.json
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "tools" / "llama.cpp-b10355" / "llama-bench.exe"
MODEL = ROOT / "models" / "gpt-oss-20b-GGUF" / "gpt-oss-20b-MXFP4.gguf"
OUT = ROOT / "results" / "gpt-oss" / "thread-sweep.json"

PLACEMENTS = [(24, 24), (24, 10)]
THREADS = [4, 6, 8, 10, 12]
REPS = "3"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for ngl, ncpu in PLACEMENTS:
        for threads in THREADS:
            cmd = [
                str(BENCH), "-m", str(MODEL),
                "-ngl", str(ngl), "--n-cpu-moe", str(ncpu),
                "-t", str(threads),
                "-p", "128", "-n", "32", "-r", REPS, "-o", "json",
            ]
            print(f"[sweep] ngl={ngl} n-cpu-moe={ncpu} -t {threads}", flush=True)
            start = time.perf_counter()
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  stdin=subprocess.DEVNULL, timeout=1800)
            wall = time.perf_counter() - start
            record = {"ngl": ngl, "n_cpu_moe": ncpu, "threads": threads,
                      "rc": proc.returncode, "wall_s": round(wall, 1)}
            if proc.returncode == 0:
                try:
                    rows = json.loads(proc.stdout)
                    gen = [r for r in rows if r.get("n_gen")]
                    pp = [r for r in rows if r.get("n_prompt")]
                    record["tg_avg_ts"] = gen[0]["avg_ts"] if gen else None
                    record["tg_stddev_ts"] = gen[0]["stddev_ts"] if gen else None
                    record["pp_avg_ts"] = pp[0]["avg_ts"] if pp else None
                    print(f"  ok: tg={record['tg_avg_ts']:.2f} tok/s", flush=True)
                except (json.JSONDecodeError, IndexError, KeyError):
                    record["status"] = "parse_error"
                    record["stdout_head"] = proc.stdout[:300]
            else:
                record["status"] = "error"
                record["stderr_tail"] = proc.stderr[-400:]
                print(f"  FAILED rc={proc.returncode}: {proc.stderr[-200:]!r}", flush=True)
            records.append(record)

    payload = {
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
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
