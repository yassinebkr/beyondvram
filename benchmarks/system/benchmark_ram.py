"""B03: repeated host-memory copy benchmark using NumPy."""

from __future__ import annotations

import argparse
import time

import numpy as np

from common import RAW_CSV, rebuild_summary, utc_now, write_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size-mib", type=int, default=512)
    parser.add_argument("--repetitions", type=int, default=7)
    args = parser.parse_args()
    nbytes = args.size_mib * 1024 * 1024
    source = np.empty(nbytes, dtype=np.uint8)
    target = np.empty_like(source)
    source.fill(0xA5)
    np.copyto(target, source)  # Fault in pages before timing.
    rows = []
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
    write_rows(RAW_CSV, rows)
    rebuild_summary()


if __name__ == "__main__":
    main()

