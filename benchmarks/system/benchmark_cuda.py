"""B04-B06 using PyTorch CUDA. Records skips rather than inventing results."""

from __future__ import annotations

import argparse
import platform

from common import RAW_CSV, rebuild_summary, skipped_row, utc_now, write_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transfer-mib", type=int, default=256)
    parser.add_argument("--transfer-repetitions", type=int, default=10)
    parser.add_argument("--matmul-size", type=int, default=4096)
    parser.add_argument("--matmul-repetitions", type=int, default=20)
    args = parser.parse_args()
    try:
        import torch
    except ImportError:
        reason = f"PyTorch is not installed for {platform.python_version()}"
        write_rows(RAW_CSV, [
            skipped_row("B04", "pageable RAM to VRAM bandwidth", reason),
            skipped_row("B05", "pinned RAM to VRAM bandwidth", reason),
            skipped_row("B06", "basic GPU compute and VRAM availability", reason),
        ])
        rebuild_summary()
        return
    if not torch.cuda.is_available():
        reason = "torch.cuda.is_available() returned false"
        write_rows(RAW_CSV, [
            skipped_row("B04", "pageable RAM to VRAM bandwidth", reason),
            skipped_row("B05", "pinned RAM to VRAM bandwidth", reason),
            skipped_row("B06", "basic GPU compute and VRAM availability", reason),
        ])
        rebuild_summary()
        return

    device = torch.device("cuda:0")
    nbytes = args.transfer_mib * 1024 * 1024
    destination = torch.empty(nbytes, dtype=torch.uint8, device=device)
    rows = []
    for benchmark_id, label, pinned in [
        ("B04", "pageable RAM to VRAM bandwidth", False),
        ("B05", "pinned RAM to VRAM bandwidth", True),
    ]:
        source = torch.empty(nbytes, dtype=torch.uint8, pin_memory=pinned)
        source.fill_(0xA5)
        destination.copy_(source, non_blocking=pinned)
        torch.cuda.synchronize()
        for repetition in range(1, args.transfer_repetitions + 1):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            destination.copy_(source, non_blocking=pinned)
            end.record()
            end.synchronize()
            seconds = start.elapsed_time(end) / 1000.0
            rows.append({
                "benchmark_id": benchmark_id, "test": label, "timestamp_utc": utc_now(),
                "repetition": repetition, "workload": "torch_tensor_copy", "buffer_bytes": nbytes,
                "chunk_bytes": nbytes, "operations": 1, "seconds": seconds,
                "value": nbytes / seconds / 1e9, "unit": "GB/s_payload", "status": "ok",
                "notes": f"pin_memory={pinned}; non_blocking={pinned}; CUDA event timing.",
            })

    props = torch.cuda.get_device_properties(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    rows.append({
        "benchmark_id": "B06", "test": "VRAM available", "timestamp_utc": utc_now(),
        "repetition": 1, "workload": props.name, "buffer_bytes": total_bytes,
        "value": free_bytes / (1024 ** 3), "unit": "GiB_free", "status": "ok",
        "notes": f"total_GiB={total_bytes / (1024 ** 3):.3f}; torch={torch.__version__}; cuda={torch.version.cuda}",
    })
    n = args.matmul_size
    left = torch.randn((n, n), dtype=torch.float16, device=device)
    right = torch.randn((n, n), dtype=torch.float16, device=device)
    torch.mm(left, right)
    torch.cuda.synchronize()
    for repetition in range(1, args.matmul_repetitions + 1):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        torch.mm(left, right)
        end.record()
        end.synchronize()
        seconds = start.elapsed_time(end) / 1000.0
        rows.append({
            "benchmark_id": "B06", "test": "basic GPU FP16 matrix multiply", "timestamp_utc": utc_now(),
            "repetition": repetition, "workload": f"torch.mm_{n}x{n}", "operations": 1,
            "seconds": seconds, "value": 2 * n ** 3 / seconds / 1e12, "unit": "TFLOP/s",
            "status": "ok", "notes": "2*n^3 operation convention; CUDA event timing.",
        })
    write_rows(RAW_CSV, rows)
    rebuild_summary()


if __name__ == "__main__":
    main()

