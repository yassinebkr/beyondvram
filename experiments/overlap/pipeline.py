"""OVR: layer-streaming overlap experiment — storage/H2D/compute scheduling."""
from __future__ import annotations

import argparse
import ctypes
import time
from pathlib import Path

import bootstrap  # noqa: F401  (adds benchmarks/system to sys.path)
from common import RAW_CSV, RESULTS_DIR, rebuild_summary, write_rows

from block_fixture import (
    DEFAULT_BLOCKS,
    FP16_LAYER_BYTES,
    INT4_LAYER_BYTES,
    expected_range_sum,
    write_fixture,
)
from rows import (
    error_row,
    makespan_row,
    memory_row,
    skipped_row,
    stage_row,
    throughput_row,
    write_timeline,
)
from storage_stage import BlockStorageReader


SIZES = {"fp16": FP16_LAYER_BYTES, "int4": INT4_LAYER_BYTES}
TIMELINE_CSV = RESULTS_DIR / "overlap_timeline.csv"


class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def process_working_set_bytes() -> int:
    """Return Windows WorkingSetSize for the current process."""
    kernel32 = ctypes.windll.kernel32
    psapi = ctypes.windll.psapi
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError()
    return counters.WorkingSetSize


def verify_checksums(sums_cpu, block_bytes: int, blocks: int) -> list[int]:
    """Return blocks whose GPU byte sums differ from the deterministic fixture."""
    return [
        block for block in range(blocks)
        if int(sums_cpu[block]) != expected_range_sum(block * block_bytes, block_bytes)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, default=RESULTS_DIR / "overlap-fixture.bin")
    parser.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--sizes", nargs="+", choices=sorted(SIZES), default=["fp16", "int4"])
    parser.add_argument("--matmul-size", type=int, default=4096)
    parser.add_argument("--keep-file", action="store_true")
    args = parser.parse_args()
    for name in ("blocks", "repetitions", "matmul_size"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main() -> None:
    args = parse_args()
    try:
        import torch
        import variants
    except ImportError as exc:
        write_rows(RAW_CSV, [skipped_row(f"dependency missing: {exc}")])
        rebuild_summary()
        return
    if not torch.cuda.is_available():
        write_rows(RAW_CSV, [skipped_row("torch.cuda.is_available() returned false")])
        rebuild_summary()
        return

    fixture_bytes = args.blocks * FP16_LAYER_BYTES
    if not (args.file.exists() and args.file.stat().st_size == fixture_bytes):
        print(f"writing fixture: {args.file} ({fixture_bytes} bytes)")
        write_fixture(args.file, fixture_bytes)

    device = torch.device("cuda:0")
    runners = {
        "sync": variants.run_sync,
        "double_buffer": variants.run_double_buffer,
        "three_stage": variants.run_three_stage,
    }
    rows: list[dict] = []
    try:
        for label in args.sizes:
            block_bytes = SIZES[label]
            workload_label = f"{label};blocks={args.blocks};matmul={args.matmul_size}"
            storage = BlockStorageReader(args.file, block_bytes)
            try:
                ctx = variants.PipelineContext(
                    device=device, blocks=args.blocks, block_bytes=block_bytes, storage=storage,
                    pinned=[torch.empty(block_bytes, dtype=torch.uint8, pin_memory=True) for _ in range(3)],
                    device_buffers=[torch.empty(block_bytes, dtype=torch.uint8, device=device) for _ in range(2)],
                    left=torch.randn(args.matmul_size, args.matmul_size, dtype=torch.float16, device=device),
                    right=torch.randn(args.matmul_size, args.matmul_size, dtype=torch.float16, device=device),
                    out=torch.empty(args.matmul_size, args.matmul_size, dtype=torch.float16, device=device),
                    sums=torch.zeros(args.blocks, dtype=torch.int64, device=device),
                    copy_stream=torch.cuda.Stream(),
                )
                ctx.gemm()
                ctx.device_buffers[0].copy_(ctx.pinned[0])
                torch.cuda.synchronize()
                for variant, runner in runners.items():
                    for repetition in range(1, args.repetitions + 1):
                        storage.reset()
                        ctx.sums.zero_()
                        torch.cuda.reset_peak_memory_stats(device)
                        start = time.perf_counter()
                        times = runner(ctx)
                        torch.cuda.synchronize()
                        makespan = time.perf_counter() - start
                        bad = verify_checksums(ctx.sums.cpu(), block_bytes, args.blocks)
                        if bad:
                            detail = f"blocks with wrong data: {bad}"
                            rows.append(error_row(f"{variant} checksum", workload_label, detail))
                            raise RuntimeError(f"checksum mismatch in {variant}/{label}: {bad}")
                        payload = args.blocks * block_bytes
                        rows.extend([
                            makespan_row(variant, workload_label, repetition, makespan, args.blocks),
                            throughput_row(variant, workload_label, repetition, payload, makespan),
                        ])
                        for stage, values in times.as_dict().items():
                            rows.append(stage_row(variant, workload_label, repetition, stage, sum(values), args.blocks, block_bytes))
                        rows.extend([
                            memory_row(variant, workload_label, repetition, "VRAM allocated",
                                       torch.cuda.memory_allocated(device), "after pass, steady state"),
                            memory_row(variant, workload_label, repetition, "process working set",
                                       process_working_set_bytes(), "psapi WorkingSetSize after pass"),
                        ])
                        write_timeline(TIMELINE_CSV, variant, workload_label, repetition, times.as_dict())
                        print(f"{label}/{variant} rep {repetition}: makespan {makespan:.3f}s "
                              f"({payload / makespan / 1e9:.3f} GB/s payload)")
            finally:
                storage.close()
    except Exception as exc:
        if not rows or rows[-1].get("status") != "error":
            rows.append(error_row("overlap pipeline", ",".join(args.sizes), repr(exc)))
        raise
    finally:
        if rows:
            write_rows(RAW_CSV, rows)
            rebuild_summary()
        if not args.keep_file:
            args.file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
