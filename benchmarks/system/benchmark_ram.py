"""B03: repeated host-memory copy benchmark using NumPy.

Progress prints per repetition. Ctrl+C writes the rows accumulated so far via
the normal write_rows/rebuild_summary path and exits with code 130.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from common import RAW_CSV, rebuild_summary, ts, utc_now, write_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size-mib", type=int, default=512)
    parser.add_argument("--repetitions", type=int, default=7)
    args = parser.parse_args()
    print(f"[{ts()}] B03 RAM memcpy: {args.size_mib} MiB buffer, "
          f"{args.repetitions} repetitions", flush=True)
    print(f"[{ts()}] output -> {RAW_CSV} (Ctrl+C writes partial rows)", flush=True)
    nbytes = args.size_mib * 1024 * 1024
    source = np.empty(nbytes, dtype=np.uint8)
    target = np.empty_like(source)
    source.fill(0xA5)
    np.copyto(target, source)  # Fault in pages before timing.
    rows = []
    interrupted = False
    try:
        for repetition in range(1, args.repetitions + 1):
            start = time.perf_counter()
            np.copyto(target, source)
            elapsed = time.perf_counter() - start
            rows.append(
                {
                    "benchmark_id": "B03",
                    "test": "RAM memcpy bandwidth",
                    "timestamp_utc": utc_now(),
                    "repetition": repetition,
                    "workload": "numpy.copyto",
                    "buffer_bytes": nbytes,
                    "chunk_bytes": nbytes,
                    "operations": 1,
                    "seconds": elapsed,
                    "value": nbytes / elapsed / 1e9,
                    "unit": "GB/s_payload",
                    "status": "ok",
                    "notes": "Payload bandwidth; approximate DRAM traffic is 2x (read plus write).",
                }
            )
            print(f"[{ts()}] [rep {repetition}/{args.repetitions}] memcpy: "
                  f"{rows[-1]['value']:.2f} GB/s in {elapsed * 1000:.1f} ms",
                  flush=True)
    except KeyboardInterrupt:
        interrupted = True
    write_rows(RAW_CSV, rows)
    rebuild_summary()
    if interrupted:
        print(f"\n[{ts()}] interrupted — partial rows ({len(rows)}) -> {RAW_CSV}",
              flush=True)
        sys.exit(130)
    print(f"[{ts()}] wrote {len(rows)} rows -> {RAW_CSV}", flush=True)


if __name__ == "__main__":
    main()

