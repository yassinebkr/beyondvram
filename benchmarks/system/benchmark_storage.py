"""B01/B02: Windows direct-I/O sequential and randomized read benchmark.

Progress prints per repetition. Ctrl+C writes the rows accumulated so far via
the normal write_rows/rebuild_summary path, still removes the fixture unless
--keep-file was passed, and exits with code 130.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import random
import sys
import time
from pathlib import Path

from common import RAW_CSV, rebuild_summary, ts, utc_now, write_rows


GENERIC_READ = 0x80000000
FILE_SHARE_READ = 0x00000001
OPEN_EXISTING = 3
FILE_FLAG_NO_BUFFERING = 0x20000000
FILE_FLAG_SEQUENTIAL_SCAN = 0x08000000
FILE_BEGIN = 0
MEM_COMMIT_RESERVE = 0x3000
PAGE_READWRITE = 0x04
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value


def create_fixture(path: Path, size_bytes: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    block = os.urandom(1024 * 1024)
    with path.open("wb", buffering=0) as handle:
        remaining = size_bytes
        while remaining:
            part = block[: min(len(block), remaining)]
            handle.write(part)
            remaining -= len(part)


class DirectReader:
    def __init__(self, path: Path, max_chunk: int):
        if os.name != "nt":
            raise RuntimeError("This direct-I/O implementation currently supports Windows only")
        self.kernel = ctypes.windll.kernel32
        self.kernel.CreateFileW.argtypes = [
            ctypes.c_wchar_p, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_void_p,
            ctypes.c_ulong, ctypes.c_ulong, ctypes.c_void_p,
        ]
        self.kernel.CreateFileW.restype = ctypes.c_void_p
        self.kernel.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulong, ctypes.c_ulong]
        self.kernel.VirtualAlloc.restype = ctypes.c_void_p
        self.kernel.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulong]
        self.kernel.VirtualFree.restype = ctypes.c_int
        self.kernel.SetFilePointerEx.argtypes = [
            ctypes.c_void_p, ctypes.c_longlong, ctypes.POINTER(ctypes.c_longlong), ctypes.c_ulong,
        ]
        self.kernel.SetFilePointerEx.restype = ctypes.c_int
        self.kernel.ReadFile.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong,
            ctypes.POINTER(ctypes.c_ulong), ctypes.c_void_p,
        ]
        self.kernel.ReadFile.restype = ctypes.c_int
        self.kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        self.kernel.CloseHandle.restype = ctypes.c_int
        self.handle = self.kernel.CreateFileW(
            str(path), GENERIC_READ, FILE_SHARE_READ, None, OPEN_EXISTING,
            FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN, None,
        )
        if self.handle == INVALID_HANDLE_VALUE:
            raise ctypes.WinError()
        self.buffer = self.kernel.VirtualAlloc(None, max_chunk, MEM_COMMIT_RESERVE, PAGE_READWRITE)
        if not self.buffer:
            raise ctypes.WinError()

    def seek(self, offset: int) -> None:
        new_position = ctypes.c_longlong()
        if not self.kernel.SetFilePointerEx(self.handle, ctypes.c_longlong(offset), ctypes.byref(new_position), FILE_BEGIN):
            raise ctypes.WinError()

    def read(self, size: int) -> None:
        read_count = ctypes.c_ulong()
        if not self.kernel.ReadFile(self.handle, self.buffer, size, ctypes.byref(read_count), None):
            raise ctypes.WinError()
        if read_count.value != size:
            raise EOFError(f"requested {size} bytes, read {read_count.value}")

    def close(self) -> None:
        self.kernel.VirtualFree(self.buffer, 0, 0x8000)
        self.kernel.CloseHandle(self.handle)


def measurement_row(test_id: str, test: str, repetition: int, workload: str, file_size: int,
                    chunk_size: int, operations: int, elapsed: float, notes: str) -> dict:
    transferred = chunk_size * operations
    return {
        "benchmark_id": test_id,
        "test": test,
        "timestamp_utc": utc_now(),
        "repetition": repetition,
        "workload": workload,
        "buffer_bytes": file_size,
        "chunk_bytes": chunk_size,
        "operations": operations,
        "seconds": elapsed,
        "value": transferred / elapsed / 1e9,
        "unit": "GB/s_payload",
        "status": "ok",
        "notes": notes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, default=Path("results/system/storage-fixture.bin"))
    parser.add_argument("--file-size-mib", type=int, default=2048)
    parser.add_argument("--sequential-chunk-mib", type=int, default=16)
    parser.add_argument("--random-total-mib", type=int, default=256)
    parser.add_argument("--random-chunks-kib", type=int, nargs="+", default=[4, 64, 1024, 16384])
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--keep-file", action="store_true")
    args = parser.parse_args()
    alignment = 4096
    file_size = args.file_size_mib * 1024 * 1024
    sequential_chunk = args.sequential_chunk_mib * 1024 * 1024
    random_chunks = [value * 1024 for value in args.random_chunks_kib]
    for value in [file_size, sequential_chunk, *random_chunks]:
        if value % alignment:
            raise ValueError("all sizes must be multiples of 4096 bytes")
    if file_size < max(sequential_chunk, *random_chunks):
        raise ValueError("fixture must be at least as large as the largest chunk")
    print(f"[{ts()}] B01/B02 storage: fixture {args.file} ({args.file_size_mib} MiB), "
          f"sequential chunk {args.sequential_chunk_mib} MiB, random chunks "
          f"{args.random_chunks_kib} KiB, {args.repetitions} repetitions", flush=True)
    print(f"[{ts()}] output -> {RAW_CSV} (Ctrl+C writes partial rows; fixture "
          f"cleanup still runs)", flush=True)
    rows = []
    interrupted = False
    try:
        print(f"[{ts()}] creating fixture ({args.file_size_mib} MiB)", flush=True)
        create_fixture(args.file, file_size)
        reader = DirectReader(args.file.resolve(), max(sequential_chunk, *random_chunks))
        try:
            seq_ops = file_size // sequential_chunk
            print(f"[{ts()}] [step 1/2] B01 sequential read: {args.repetitions} reps "
                  f"x {seq_ops} ops of {args.sequential_chunk_mib} MiB", flush=True)
            for repetition in range(1, args.repetitions + 1):
                reader.seek(0)
                start = time.perf_counter()
                for _ in range(seq_ops):
                    reader.read(sequential_chunk)
                elapsed = time.perf_counter() - start
                rows.append(measurement_row(
                    "B01", "NVMe sequential read bandwidth", repetition, "direct_io_sequential",
                    file_size, sequential_chunk, seq_ops, elapsed,
                    "Windows FILE_FLAG_NO_BUFFERING; payload uses decimal GB/s.",
                ))
                print(f"[{ts()}] [step 1/2] B01 rep {repetition}/{args.repetitions}: "
                      f"{rows[-1]['value']:.3f} GB/s in {elapsed:.2f}s", flush=True)
            print(f"[{ts()}] [step 2/2] B02 random reads: {len(random_chunks)} chunk "
                  f"sizes x {args.repetitions} reps", flush=True)
            for chunk in random_chunks:
                operations = max(1, args.random_total_mib * 1024 * 1024 // chunk)
                slot_count = file_size // chunk
                for repetition in range(1, args.repetitions + 1):
                    rng = random.Random(0xBEEFBEEF + repetition * 1009 + chunk)
                    offsets = [rng.randrange(slot_count) * chunk for _ in range(operations)]
                    start = time.perf_counter()
                    for offset in offsets:
                        reader.seek(offset)
                        reader.read(chunk)
                    elapsed = time.perf_counter() - start
                    rows.append(measurement_row(
                        "B02", "NVMe random/chunked read behavior", repetition,
                        f"direct_io_random_{chunk // 1024}KiB", file_size, chunk, operations, elapsed,
                        "Deterministic random offsets; Windows FILE_FLAG_NO_BUFFERING.",
                    ))
                    print(f"[{ts()}] [step 2/2] B02 {chunk // 1024} KiB rep "
                          f"{repetition}/{args.repetitions}: {rows[-1]['value']:.3f} GB/s "
                          f"in {elapsed:.2f}s", flush=True)
        finally:
            reader.close()
    except KeyboardInterrupt:
        interrupted = True
    finally:
        if not args.keep_file:
            args.file.unlink(missing_ok=True)
    write_rows(RAW_CSV, rows)
    rebuild_summary()
    if interrupted:
        print(f"\n[{ts()}] interrupted — partial rows ({len(rows)}) -> {RAW_CSV}",
              flush=True)
        sys.exit(130)
    print(f"[{ts()}] wrote {len(rows)} rows -> {RAW_CSV}", flush=True)


if __name__ == "__main__":
    main()
