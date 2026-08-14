"""Pack, stream, execute, and validate one Qwen3-8B decoder layer.

Progress banners are timestamped and flushed; Ctrl+C writes the partial result
rows collected so far and exits with code 130.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import bootstrap  # noqa: F401  (adds benchmarks/system to sys.path)
from benchmark_storage import DirectReader
from common import RAW_CSV, rebuild_summary, ts, utc_now, write_rows


ALIGNMENT = 4096
MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class TensorSpec:
    """One tensor's byte range in a direct-I/O-friendly layer package."""

    name: str
    shape: tuple[int, ...]
    offset: int
    nbytes: int


def align_up(value: int, alignment: int = ALIGNMENT) -> int:
    """Return the next aligned byte position."""
    return (value + alignment - 1) // alignment * alignment


def local_layer_names(model_dir: Path, layer_index: int) -> list[str]:
    """Return sorted state-dict names belonging to a single decoder layer."""
    index_path = model_dir / "model.safetensors.index.json"
    with index_path.open(encoding="utf-8") as handle:
        weight_map = json.load(handle)["weight_map"]
    prefix = f"model.layers.{layer_index}."
    names = sorted(name for name in weight_map if name.startswith(prefix))
    if not names:
        raise ValueError(f"no tensors found for layer {layer_index} in {index_path}")
    return names


def pack_layer(model_dir: Path, output_dir: Path, layer_index: int) -> Path:
    """Create aligned BF16 raw bytes plus a manifest from local Safetensors shards."""
    from safetensors import safe_open

    config_path = model_dir / "config.json"
    index_path = model_dir / "model.safetensors.index.json"
    if not config_path.exists() or not index_path.exists():
        raise FileNotFoundError("model directory must contain config.json and model.safetensors.index.json")
    with index_path.open(encoding="utf-8") as handle:
        weight_map = json.load(handle)["weight_map"]
    names = local_layer_names(model_dir, layer_index)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / f"qwen3-8b-layer-{layer_index:02d}.bf16.bin"
    specs: list[TensorSpec] = []
    offset = 0
    with data_path.open("wb", buffering=0) as output:
        for name in names:
            shard = model_dir / weight_map[name]
            with safe_open(str(shard), framework="pt", device="cpu") as handle:
                tensor = handle.get_tensor(name).contiguous()
            if str(tensor.dtype) != "torch.bfloat16":
                raise ValueError(f"{name} has {tensor.dtype}; this first harness requires BF16 checkpoint tensors")
            offset = align_up(offset)
            output.seek(offset)
            output.write(tensor.view(torch.uint8).numpy().tobytes())
            specs.append(TensorSpec(name.removeprefix(f"model.layers.{layer_index}."), tuple(tensor.shape), offset, tensor.nbytes))
            offset += tensor.nbytes
        output.truncate(align_up(offset))
    manifest = {
        "format": "beyondvram-real-layer-v1",
        "source_model": str(model_dir),
        "layer_index": layer_index,
        "dtype": "bfloat16",
        "alignment_bytes": ALIGNMENT,
        "data_file": data_path.name,
        "payload_bytes": sum(spec.nbytes for spec in specs),
        "file_bytes": data_path.stat().st_size,
        "tensors": [spec.__dict__ for spec in specs],
    }
    (output_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_dir / MANIFEST_NAME


def load_manifest(path: Path) -> tuple[dict, list[TensorSpec]]:
    """Load and validate an aligned layer package manifest."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("format") != "beyondvram-real-layer-v1" or data.get("alignment_bytes") != ALIGNMENT:
        raise ValueError(f"unsupported manifest: {path}")
    specs = [TensorSpec(item["name"], tuple(item["shape"]), item["offset"], item["nbytes"]) for item in data["tensors"]]
    for spec in specs:
        if spec.offset % ALIGNMENT or spec.nbytes <= 0:
            raise ValueError(f"unaligned or empty tensor entry: {spec.name}")
    return data, specs


def result_row(test: str, workload: str, repetition: int, seconds: float, value: float, unit: str,
               *, buffer_bytes: int = 0, operations: int = 1, notes: str = "") -> dict:
    return {"benchmark_id": "RLS", "test": test, "timestamp_utc": utc_now(), "repetition": repetition,
            "workload": workload, "buffer_bytes": buffer_bytes, "operations": operations,
            "seconds": seconds, "value": value, "unit": unit, "status": "ok", "notes": notes}


def load_reference_state(model_dir: Path, layer_index: int):
    """Read only one layer from Safetensors into CPU tensors for the trusted reference."""
    from safetensors import safe_open

    with (model_dir / "model.safetensors.index.json").open(encoding="utf-8") as handle:
        weight_map = json.load(handle)["weight_map"]
    state = {}
    prefix = f"model.layers.{layer_index}."
    for full_name in local_layer_names(model_dir, layer_index):
        with safe_open(str(model_dir / weight_map[full_name]), framework="pt", device="cpu") as handle:
            state[full_name.removeprefix(prefix)] = handle.get_tensor(full_name)
    return state


def build_layer(config_path: Path, layer_index: int, device: str):
    """Construct the official Transformers Qwen3 layer without loading whole-model weights."""
    from transformers import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

    config = Qwen3Config.from_json_file(str(config_path))
    config._attn_implementation = "eager"
    return Qwen3DecoderLayer(config, layer_index).to(device=device, dtype=torch.bfloat16), config


def assign_buffer_views(layer, device_bytes, specs: list[TensorSpec]) -> None:
    """Make layer parameters view the streamed device package; no per-tensor device copies."""
    params = dict(layer.named_parameters())
    raw_words = device_bytes.view(torch.bfloat16)
    for spec in specs:
        parameter = params.get(spec.name)
        if parameter is None:
            raise KeyError(f"manifest tensor not in Qwen layer: {spec.name}")
        start = spec.offset // 2
        count = spec.nbytes // 2
        parameter.data = raw_words.narrow(0, start, count).view(spec.shape)


def run_layer(layer, config, hidden_states, position_ids):
    """Run the official layer with a deterministic one-token input and no KV cache."""
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

    rotary = Qwen3RotaryEmbedding(config=config, device=hidden_states.device).to(hidden_states.device)
    positions = rotary(hidden_states, position_ids)
    return layer(hidden_states, attention_mask=None, position_ids=position_ids,
                 use_cache=False, position_embeddings=positions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True, help="local Qwen/Qwen3-8B Safetensors directory")
    parser.add_argument("--package-dir", type=Path, default=Path("results/real_layer"))
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--pack-only", action="store_true")
    args = parser.parse_args()
    if args.layer < 0 or args.sequence_length < 1 or args.repetitions < 1:
        raise ValueError("layer >= 0, sequence-length >= 1, and repetitions >= 1 are required")
    manifest_path = args.package_dir / MANIFEST_NAME
    rows: list[dict] = []
    reader = None
    try:
        if not manifest_path.exists():
            print(f"[{ts()}] packing layer {args.layer} from {args.model_dir} "
                  f"-> {args.package_dir}", flush=True)
            manifest_path = pack_layer(args.model_dir, args.package_dir, args.layer)
        if args.pack_only:
            print(f"wrote {manifest_path}", flush=True)
            return
        manifest, specs = load_manifest(manifest_path)
        if manifest["layer_index"] != args.layer:
            raise ValueError("package layer does not match --layer")
        data_path = args.package_dir / manifest["data_file"]
        payload = int(manifest["payload_bytes"])
        file_bytes = int(manifest["file_bytes"])
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the streamed execution measurement")
        print(f"[{ts()}] RLS layer stream: layer={args.layer} seq={args.sequence_length} "
              f"repetitions={args.repetitions} package={data_path.name} "
              f"({file_bytes} bytes, payload {payload} bytes)", flush=True)
        print(f"[{ts()}] rows -> {RAW_CSV} (Ctrl+C writes partial results)", flush=True)
        reference, config = build_layer(args.model_dir / "config.json", args.layer, "cuda")
        reference.load_state_dict(load_reference_state(args.model_dir, args.layer), strict=True)
        reference.eval()
        streamed, _ = build_layer(args.model_dir / "config.json", args.layer, "cuda")
        streamed.eval()
        pinned = torch.empty(file_bytes, dtype=torch.uint8, pin_memory=True)
        device_bytes = torch.empty(file_bytes, dtype=torch.uint8, device="cuda")
        reader = DirectReader(data_path.resolve(), file_bytes)
        copy_stream = torch.cuda.Stream()
        generator = torch.Generator(device="cuda").manual_seed(0xB30A0D)
        hidden = torch.randn((1, args.sequence_length, config.hidden_size), device="cuda", dtype=torch.bfloat16,
                             generator=generator)
        position_ids = torch.arange(args.sequence_length, device="cuda").unsqueeze(0)
        with torch.inference_mode():
            reference_output = run_layer(reference, config, hidden, position_ids).detach()
        for repetition in range(1, args.repetitions + 1):
            print(f"[{ts()}] [step {repetition}/{args.repetitions}] direct read -> "
                  f"RAM staging -> pinned H2D -> layer compute", flush=True)
            read_start = time.perf_counter()
            reader.seek(0)
            reader.read(file_bytes)
            read_seconds = time.perf_counter() - read_start
            source = (ctypes.c_char * file_bytes).from_address(reader.buffer)
            copy_start = time.perf_counter()
            pinned.numpy()[:] = np.frombuffer(source, dtype=np.uint8)
            stage_seconds = time.perf_counter() - copy_start
            start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(copy_stream):
                start_event.record(copy_stream)
                device_bytes.copy_(pinned, non_blocking=True)
                end_event.record(copy_stream)
            end_event.synchronize()
            h2d_seconds = start_event.elapsed_time(end_event) / 1000.0
            assign_buffer_views(streamed, device_bytes, specs)
            start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start_event.record()
            with torch.inference_mode():
                output = run_layer(streamed, config, hidden, position_ids)
            end_event.record()
            end_event.synchronize()
            compute_seconds = start_event.elapsed_time(end_event) / 1000.0
            max_abs = (output.float() - reference_output.float()).abs().max().item()
            passed = math.isfinite(max_abs) and max_abs == 0.0
            workload = f"qwen3-8b;layer={args.layer};seq={args.sequence_length};file={file_bytes}"
            rows.extend([
                result_row("direct read", workload, repetition, read_seconds, read_seconds, "s", buffer_bytes=file_bytes, notes="FILE_FLAG_NO_BUFFERING aligned layer package"),
                result_row("RAM to pinned staging", workload, repetition, stage_seconds, stage_seconds, "s", buffer_bytes=file_bytes),
                result_row("pinned H2D", workload, repetition, h2d_seconds, h2d_seconds, "s", buffer_bytes=file_bytes),
                result_row("streamed Qwen decoder layer compute", workload, repetition, compute_seconds, compute_seconds, "s", operations=args.sequence_length),
                result_row("reference versus streamed max abs error", workload, repetition, 0.0, max_abs, "abs_error", notes="exact BF16 byte package; expected zero"),
            ])
            print(f"[{ts()}] rep {repetition}: read={read_seconds:.3f}s stage={stage_seconds:.3f}s h2d={h2d_seconds:.3f}s compute={compute_seconds:.3f}s max_abs={max_abs:.6g}", flush=True)
            if not passed:
                raise AssertionError(f"streamed output differs from reference: max_abs={max_abs}")
    except KeyboardInterrupt:
        print(f"\n[{ts()}] interrupted — writing partial results ({len(rows)} rows) "
              f"-> {RAW_CSV}", flush=True)
        sys.exit(130)
    finally:
        if reader is not None:
            reader.close()
        if rows:
            write_rows(RAW_CSV, rows)
            rebuild_summary()


if __name__ == "__main__":
    main()
