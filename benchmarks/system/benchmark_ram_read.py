"""B07: host-memory pure-read bandwidth benchmark using NumPy.

B03 measures a copy (read + write). Expert-style inference workloads are
read-dominated, so this benchmark measures a pure sequential read stream
(np.sum over a large buffer), single-threaded and multi-threaded, to give
the effective-bandwidth roofline a measured denominator.
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from common import RAW_CSV, rebuild_summary, utc_now, write_rows


def read_sum(source: np.ndarray, threads: int) -> None:
    if threads <= 1:
        np.sum(source)
        return
    slices = np.array_split(source, threads)
    with ThreadPoolExecutor(max_workers=threads) as pool:
        list(pool.map(np.sum, slices))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size-mib", type=int, default=512)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 1,
                        help="thread count for the multi-threaded workload (default: logical cores)")
    args = parser.parse_args()
    nbytes = args.size_mib * 1024 * 1024
    # float64 elements: one SIMD add per 8 bytes keeps the loop memory-bound;
    # a uint8 sum is compute-bound per byte and would measure the kernel, not DRAM.
    source = np.empty(nbytes // 8, dtype=np.float64)
    source.fill(1.0)
    read_sum(source, 1)
    read_sum(source, args.threads)  # Fault in pages before timing.
    rows = []
    for repetition in range(1, args.repetitions + 1):
        for workload, threads in (("numpy.sum.f64.st", 1), (f"numpy.sum.f64.mt{args.threads}", args.threads)):
            start = time.perf_counter()
            read_sum(source, threads)
            elapsed = time.perf_counter() - start
            rows.append(
                {
                    "benchmark_id": "B07",
                    "test": "RAM read bandwidth",
                    "timestamp_utc": utc_now(),
                    "repetition": repetition,
                    "workload": workload,
                    "buffer_bytes": nbytes,
                    "chunk_bytes": nbytes,
                    "operations": 1,
                    "seconds": elapsed,
                    "value": nbytes / elapsed / 1e9,
                    "unit": "GB/s_payload",
                    "status": "ok",
                    "notes": "Pure sequential read (np.sum over float64); no write stream. "
                             "Denominator for read-dominated (expert-streaming) rooflines.",
                }
            )
    write_rows(RAW_CSV, rows)
    rebuild_summary()


if __name__ == "__main__":
    main()
