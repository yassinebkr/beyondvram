"""Validate serial and overlapped streaming of Qwen3-8B decoder layers 0--2."""
from __future__ import annotations

import argparse
import ctypes
import math
import threading
import time
from pathlib import Path

import numpy as np
import torch

import bootstrap  # noqa: F401
from benchmark_storage import DirectReader
from common import RAW_CSV, rebuild_summary, write_rows
from qwen_layer_stream import (
    MANIFEST_NAME, assign_buffer_views, build_layer, load_manifest, load_reference_state,
    pack_layer, result_row, run_layer,
)


def cuda_seconds(operation) -> float:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    operation()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / 1000.0


def ensure_packages(model_dir: Path, package_root: Path, layers: list[int]):
    packages = []
    for index in layers:
        directory = package_root / f"layer{index:02d}"
        manifest_path = directory / MANIFEST_NAME
        if not manifest_path.exists():
            pack_layer(model_dir, directory, index)
        manifest, specs = load_manifest(manifest_path)
        packages.append((directory / manifest["data_file"], int(manifest["file_bytes"]), specs))
    return packages


def reference_chain(model_dir: Path, layers: list[int], hidden, position_ids):
    refs = []
    config = None
    for index in layers:
        layer, config = build_layer(model_dir / "config.json", index, "cuda")
        layer.load_state_dict(load_reference_state(model_dir, index), strict=True)
        layer.eval()
        refs.append(layer)
    with torch.inference_mode():
        output = hidden
        for layer in refs:
            output = run_layer(layer, config, output, position_ids)
    return output.detach(), config


def run_serial(packages, layers, config, hidden, position_ids):
    pinned = torch.empty(max(item[1] for item in packages), dtype=torch.uint8, pin_memory=True)
    device = torch.empty_like(pinned, device="cuda")
    streamed = STREAMED
    readers = [DirectReader(path.resolve(), size) for path, size, _ in packages]
    times = dict(read=0.0, stage=0.0, h2d=0.0, compute=0.0)
    output = hidden
    try:
        for layer, reader, (_, size, specs) in zip(streamed, readers, packages):
            start = time.perf_counter(); reader.seek(0); reader.read(size); times["read"] += time.perf_counter() - start
            start = time.perf_counter(); pinned[:size].numpy()[:] = np.frombuffer((ctypes.c_char * size).from_address(reader.buffer), dtype=np.uint8); times["stage"] += time.perf_counter() - start
            times["h2d"] += cuda_seconds(lambda: device[:size].copy_(pinned[:size], non_blocking=True))
            assign_buffer_views(layer, device, specs)
            holder = [output]
            times["compute"] += cuda_seconds(lambda: holder.__setitem__(0, run_layer(layer, config, holder[0], position_ids)))
            output = holder[0]
    finally:
        for reader in readers: reader.close()
    return output, times


def run_overlapped(packages, layers, config, hidden, position_ids):
    """One storage worker overlaps read/stage of layer N+1 with GPU compute N."""
    maximum = max(item[1] for item in packages)
    pinned = [torch.empty(maximum, dtype=torch.uint8, pin_memory=True) for _ in packages]
    device = [torch.empty(maximum, dtype=torch.uint8, device="cuda") for _ in range(2)]
    streamed = STREAMED
    ready = [threading.Event() for _ in packages]
    storage_times = [0.0] * len(packages); stage_times = [0.0] * len(packages)
    errors = []
    def worker():
        try:
            for i, (path, size, _) in enumerate(packages):
                reader = DirectReader(path.resolve(), size)
                try:
                    start = time.perf_counter(); reader.read(size); storage_times[i] = time.perf_counter() - start
                    start = time.perf_counter(); pinned[i][:size].numpy()[:] = np.frombuffer((ctypes.c_char * size).from_address(reader.buffer), dtype=np.uint8); stage_times[i] = time.perf_counter() - start
                finally: reader.close()
                ready[i].set()
        except BaseException as error:
            errors.append(error)
            for event in ready: event.set()
    thread = threading.Thread(target=worker, daemon=True); thread.start()
    h2d = compute = 0.0; output = hidden
    for i, (layer, (_, size, specs)) in enumerate(zip(streamed, packages)):
        ready[i].wait()
        if errors: raise RuntimeError("storage worker failed") from errors[0]
        h2d += cuda_seconds(lambda: device[i % 2][:size].copy_(pinned[i][:size], non_blocking=True))
        assign_buffer_views(layer, device[i % 2], specs)
        holder = [output]
        compute += cuda_seconds(lambda: holder.__setitem__(0, run_layer(layer, config, holder[0], position_ids)))
        output = holder[0]
    thread.join()
    return output, {"read": sum(storage_times), "stage": sum(stage_times), "h2d": h2d, "compute": compute}


def main() -> None:
    global CONFIG_PATH, STREAMED
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, default=Path("results/real_layer/three_layers"))
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.repetitions < 1: raise ValueError("repetitions must be positive")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    layers = [0, 1, 2]; CONFIG_PATH = args.model_dir / "config.json"
    packages = ensure_packages(args.model_dir, args.package_root, layers)
    STREAMED = [build_layer(CONFIG_PATH, index, "cuda")[0].eval() for index in layers]
    hidden = torch.randn((1, 1, 4096), device="cuda", dtype=torch.bfloat16, generator=torch.Generator(device="cuda").manual_seed(0xB30A0D))
    positions = torch.zeros((1, 1), dtype=torch.long, device="cuda")
    reference, config = reference_chain(args.model_dir, layers, hidden, positions)
    rows = []
    for variant, runner in (("serial", run_serial), ("storage_overlap", run_overlapped)):
        for repetition in range(1, args.repetitions + 1):
            start = time.perf_counter(); output, times = runner(packages, layers, config, hidden, positions); makespan = time.perf_counter() - start
            error = (output.float() - reference.float()).abs().max().item()
            workload = "qwen3-8b;layers=0-2;seq=1;scheduler=v2"
            for stage, seconds in times.items(): rows.append(result_row(f"three-layer {variant} {stage}", workload, repetition, seconds, seconds, "s", operations=3))
            rows += [result_row(f"three-layer {variant} makespan", workload, repetition, makespan, makespan, "s", operations=3), result_row(f"three-layer {variant} max abs error", workload, repetition, 0.0, error, "abs_error")]
            print(f"{variant} rep {repetition}: makespan={makespan:.3f}s max_abs={error:.6g}")
            if not math.isfinite(error) or error != 0: raise AssertionError(f"correctness failed: {error}")
    write_rows(RAW_CSV, rows); rebuild_summary()


if __name__ == "__main__": main()
