# Layer-Streaming Overlap Experiment (OVR) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether storage-read, H2D-transfer and GPU-compute stages of a layer-streaming pipeline can overlap on this machine, using synthetic Qwen3-8B-layer-sized blocks, three scheduling variants, and checksum-verified data movement.

**Architecture:** A new self-contained experiment in `experiments/overlap/` that reuses the existing Windows direct-I/O reader (`benchmarks/system/benchmark_storage.py:DirectReader`) and result machinery (`benchmarks/system/common.py`) via a `sys.path` bootstrap module. Three scheduling variants — synchronous, double-buffered H2D+compute, and three-stage storage+H2D+compute — move 36 deterministic blocks through NVMe → aligned staging → pinned ring → VRAM ping-pong → synthetic GEMM, recording per-stage and end-to-end timings. Checksums are verified asynchronously on GPU against analytically-computed expected values.

**Tech Stack:** Python 3.14 `.venv` (torch 2.13.0+cu126, NumPy, pytest), Windows `ctypes` direct I/O, CUDA streams/events via PyTorch.

**Repo note:** the repository is **not under version control** (no `.git`). Commit steps are intentionally absent; if the user initializes git, commit after each task. Checkpoint instead: verify each task's outputs before moving on.

**Measurement discipline (from `AGENTS.md`, binding):** record, don't invent; keep every repetition; skipped/error rows instead of fabricated values; `GB/s_payload` semantics; direct I/O only for storage numbers; no overlap claims without per-stage + end-to-end timing (this experiment is exactly that gate).

---

## Background the engineer needs

- Run everything from the repository root `C:/Users/yassi/Documents/code/BeyondVram` with `.venv\Scripts\python.exe`.
- `benchmarks/system/benchmark_storage.py` provides `DirectReader(path, max_chunk)` with `.seek(offset)`, `.read(size)`, `.close()`, and the raw aligned buffer pointer `.buffer`. It uses `FILE_FLAG_NO_BUFFERING`; all offsets/sizes must be 4096-byte aligned.
- `benchmarks/system/common.py` provides `write_rows(path, rows)`, `rebuild_summary()`, `utc_now()`, `RAW_CSV`, `RESULTS_DIR`, and the fixed CSV `FIELDS` list. New rows must only use keys from `FIELDS`.
- Block sizes (from `docs/model-selection.md`): one Qwen3-8B FP16 layer = 193,601,536 params × 2 bytes = **387,203,072 bytes** (naturally 4096-aligned); the INT4 variant uses **113 MiB = 118,489,088 bytes** (aligned). Default pass = 36 blocks.
- Measured medians to compare against: B01 NVMe sequential 2.526 GB/s, B03 RAM copy 17.783 GB/s, B05 pinned H2D 26.061 GB/s, B06 GEMM ~35 TFLOP/s median.
- GPU runs in **WDDM** mode; cross-stream overlap works but with scheduling overhead. Record as a confounder, not a blocker.
- VRAM: 2 device buffers × 387 MB + GEMM operands ≈ 0.9 GB — fits the ~6.7 GiB observed free. Pinned ring: 3 × 387 MB host RAM — fits.
- Checksum design: the fixture is written in 4096-byte cells; cell `c` is filled with byte value `c % 251`. Any aligned byte range's expected sum is computable analytically (closed form), so GPU-side sums verify that the right bytes landed in VRAM without a CPU reference copy.
- Hypotheses (to test, not to assume): `sync` makespan ≈ sum of stage times; `double_buffer` and `three_stage` hide H2D+compute under storage, so makespans approach the storage+staging-copy bound; the staging copy (aligned buffer → pinned, ~22 ms/block FP16 at B03 rates) shows up as a real cost that would later justify a native direct-to-pinned reader (path C).

## File structure

```
experiments/overlap/
  bootstrap.py       sys.path insert for benchmarks/system (side-effect module)
  block_fixture.py   cell pattern, analytic checksum, fixture writer, block-size constants
  rows.py            CSV row builders + timeline writer (reuses common.py)
  storage_stage.py   BlockStorageReader: DirectReader -> pinned tensor, split read/copy timing
  variants.py        PipelineContext, StageTimes, run_sync, run_double_buffer, run_three_stage
  pipeline.py        CLI orchestration, checksum verification, memory capture, result writing
  README.md          purpose, design, run instructions, confounders
tests/
  conftest.py        sys.path setup for repo root + experiments/overlap
  test_overlap_logic.py   pure-logic tests (no GPU required for most)
results/system/
  overlap_timeline.csv    per-block per-stage durations (appended by runs)
```

---

### Task 1: Scaffold, fixture module, and its tests

**Files:**
- Create: `experiments/overlap/__init__.py` (NOT needed — flat scripts + conftest path setup)
- Create: `experiments/overlap/bootstrap.py`
- Create: `experiments/overlap/block_fixture.py`
- Create: `tests/conftest.py`
- Test: `tests/test_overlap_logic.py`

- [ ] **Step 1: Install pytest into the venv**

Run: `uv pip install --python .venv/Scripts/python.exe pytest`
Expected: `+ pytest==...` installed.

- [ ] **Step 2: Write the failing tests**

Create `tests/conftest.py`:

```python
"""Pytest path setup: repo root and the overlap experiment directory."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for candidate in (ROOT, ROOT / "experiments" / "overlap"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
```

Create `tests/test_overlap_logic.py` (initial portion; later tasks append more tests):

```python
"""Pure-logic tests for the overlap pipeline experiment (no GPU required)."""
from __future__ import annotations

import numpy as np
import pytest

from block_fixture import (
    CELL_BYTES,
    FILL_PERIOD,
    FP16_LAYER_BYTES,
    INT4_LAYER_BYTES,
    cell_fill_byte,
    expected_range_sum,
)


def test_block_size_constants_are_aligned():
    assert FP16_LAYER_BYTES == 193_601_536 * 2
    assert FP16_LAYER_BYTES % CELL_BYTES == 0
    assert INT4_LAYER_BYTES == 113 * 1024 * 1024
    assert INT4_LAYER_BYTES % CELL_BYTES == 0


def test_cell_fill_pattern_is_deterministic():
    assert cell_fill_byte(0) == 0
    assert cell_fill_byte(250) == 250
    assert cell_fill_byte(251) == 0
    assert cell_fill_byte(502) == 0


def test_expected_range_sum_matches_bruteforce():
    cells = (np.arange(600, dtype=np.int64) % FILL_PERIOD).astype(np.uint8)
    raw = np.repeat(cells, CELL_BYTES)
    for offset_cells, size_cells in [(0, 600), (3, 251), (100, 77)]:
        offset, size = offset_cells * CELL_BYTES, size_cells * CELL_BYTES
        assert expected_range_sum(offset, size) == int(raw[offset:offset + size].sum(dtype=np.int64))


def test_expected_range_sum_rejects_misalignment():
    with pytest.raises(ValueError):
        expected_range_sum(1, CELL_BYTES)
    with pytest.raises(ValueError):
        expected_range_sum(0, CELL_BYTES - 1)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'block_fixture'`.

- [ ] **Step 4: Implement bootstrap and fixture module**

Create `experiments/overlap/bootstrap.py`:

```python
"""Add benchmarks/system to sys.path so experiments reuse the existing helpers."""
from __future__ import annotations

import sys
from pathlib import Path

SYSTEM_BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks" / "system"
if str(SYSTEM_BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(SYSTEM_BENCHMARKS))
```

Create `experiments/overlap/block_fixture.py`:

```python
"""Deterministic fixture content, analytic checksums, and block-size constants."""
from __future__ import annotations

from pathlib import Path

import numpy as np

CELL_BYTES = 4096       # direct-I/O alignment unit; content varies per cell
FILL_PERIOD = 251       # prime; cell c is filled with byte value c % FILL_PERIOD
SPAN_CELLS = 4096       # one write span = 4096 cells = 16 MiB

# Qwen3-8B per-layer parameter count (docs/model-selection.md).
LAYER_PARAMS = 193_601_536
FP16_LAYER_BYTES = LAYER_PARAMS * 2   # 387_203_072 bytes, 4096-aligned
INT4_LAYER_BYTES = 113 * 1024 * 1024  # 118_489_088 bytes, 4096-aligned
DEFAULT_BLOCKS = 36


def cell_fill_byte(cell_index: int) -> int:
    return cell_index % FILL_PERIOD


def expected_range_sum(offset_bytes: int, size_bytes: int) -> int:
    """Closed-form sum of all fixture bytes in an aligned range."""
    if offset_bytes % CELL_BYTES or size_bytes % CELL_BYTES:
        raise ValueError("offset and size must be 4096-byte aligned")
    first_cell = offset_bytes // CELL_BYTES
    cells = size_bytes // CELL_BYTES
    base = first_cell % FILL_PERIOD
    periods, remainder = divmod(cells, FILL_PERIOD)
    total = periods * (FILL_PERIOD * (FILL_PERIOD - 1) // 2)
    total += sum((base + i) % FILL_PERIOD for i in range(remainder))
    return total * CELL_BYTES


def write_fixture(path: Path, size_bytes: int) -> None:
    """Write the deterministic fixture: cell c filled with byte c % FILL_PERIOD."""
    if size_bytes % CELL_BYTES:
        raise ValueError("fixture size must be 4096-byte aligned")
    path.parent.mkdir(parents=True, exist_ok=True)
    total_cells = size_bytes // CELL_BYTES
    with path.open("wb", buffering=0) as handle:
        for start in range(0, total_cells, SPAN_CELLS):
            count = min(SPAN_CELLS, total_cells - start)
            residues = (np.arange(start, start + count, dtype=np.int64) % FILL_PERIOD)
            span = np.repeat(residues.astype(np.uint8), CELL_BYTES)
            handle.write(memoryview(span))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests -v`
Expected: 4 passed.

---

### Task 2: Row builders and timeline writer

**Files:**
- Create: `experiments/overlap/rows.py`
- Test: `tests/test_overlap_logic.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_overlap_logic.py`:

```python
def test_rows_use_known_fields_and_units():
    from common import FIELDS
    from rows import makespan_row, memory_row, stage_row, throughput_row

    stage = stage_row("sync", "fp16", 1, "read", 1.5, 36, FP16_LAYER_BYTES)
    assert set(stage) <= set(FIELDS)
    assert stage["benchmark_id"] == "OVR"
    assert stage["unit"] == "s" and stage["status"] == "ok"
    assert makespan_row("sync", "fp16", 1, 7.0, 36)["value"] == 7.0
    throughput = throughput_row("sync", "fp16", 1, 36 * FP16_LAYER_BYTES, 7.0)
    assert throughput["unit"] == "GB/s_payload"
    assert throughput["value"] == pytest.approx(36 * FP16_LAYER_BYTES / 7.0 / 1e9)
    assert memory_row("sync", "fp16", "process working set", 1234, "x")["unit"] == "bytes"


def test_timeline_writer_appends_rows(tmp_path):
    from rows import write_timeline

    target = tmp_path / "timeline.csv"
    write_timeline(target, "sync", "fp16", 1, {"read": [0.1, 0.2], "h2d": [0.01, 0.01]})
    write_timeline(target, "sync", "fp16", 2, {"read": [0.3]})
    lines = target.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "variant,block_label,repetition,block,stage,seconds"
    assert len(lines) == 1 + 4 + 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rows'`.

- [ ] **Step 3: Implement rows.py**

Create `experiments/overlap/rows.py`:

```python
"""CSV row builders and the per-block timeline writer for the OVR experiment."""
from __future__ import annotations

import csv
from pathlib import Path

import bootstrap  # noqa: F401  (sys.path side effect)
from common import utc_now

BENCHMARK_ID = "OVR"


def _base(test: str, repetition: int, workload: str, seconds, value, unit: str, notes: str) -> dict:
    return {
        "benchmark_id": BENCHMARK_ID,
        "test": test,
        "timestamp_utc": utc_now(),
        "repetition": repetition,
        "workload": workload,
        "seconds": seconds,
        "value": value,
        "unit": unit,
        "status": "ok",
        "notes": notes,
    }


def stage_row(variant: str, block_label: str, repetition: int, stage: str,
              seconds: float, blocks: int, block_bytes: int) -> dict:
    return _base(f"{variant} stage {stage}", repetition, block_label, seconds, seconds, "s",
                 f"sum over {blocks} blocks of {block_bytes} bytes; stage={stage}")


def makespan_row(variant: str, block_label: str, repetition: int,
                 seconds: float, blocks: int) -> dict:
    return _base(f"{variant} makespan", repetition, block_label, seconds, seconds, "s",
                 f"end-to-end wall time for {blocks} blocks")


def throughput_row(variant: str, block_label: str, repetition: int,
                   payload_bytes: int, seconds: float) -> dict:
    return _base(f"{variant} payload throughput", repetition, block_label, seconds,
                 payload_bytes / seconds / 1e9, "GB/s_payload",
                 "payload bytes over end-to-end makespan")


def memory_row(variant: str, block_label: str, name: str, value_bytes: int, notes: str) -> dict:
    return _base(f"{variant} {name}", 1, block_label, "", value_bytes, "bytes", notes)


def error_row(test: str, workload: str, reason: str) -> dict:
    return {"benchmark_id": BENCHMARK_ID, "test": test, "timestamp_utc": utc_now(),
            "workload": workload, "status": "error", "notes": reason}


def skipped_row(reason: str) -> dict:
    return {"benchmark_id": BENCHMARK_ID, "test": "overlap pipeline",
            "timestamp_utc": utc_now(), "status": "skipped", "notes": reason}


def write_timeline(path: Path, variant: str, block_label: str, repetition: int,
                   stage_times: dict[str, list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(["variant", "block_label", "repetition", "block", "stage", "seconds"])
        for stage, values in stage_times.items():
            for block, seconds in enumerate(values):
                writer.writerow([variant, block_label, repetition, block, stage, f"{seconds:.9f}"])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests -v`
Expected: 6 passed.

---

### Task 3: Storage stage adapter

**Files:**
- Create: `experiments/overlap/storage_stage.py`
- Test: `tests/test_overlap_logic.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_overlap_logic.py` (add `import os` at the top of the file):

```python
@pytest.mark.skipif(os.name != "nt", reason="direct-I/O reader is Windows-only")
def test_block_storage_reader_reads_expected_content(tmp_path):
    from block_fixture import write_fixture
    from storage_stage import BlockStorageReader

    block = CELL_BYTES * FILL_PERIOD * 2  # two fill periods per block
    path = tmp_path / "fixture.bin"
    write_fixture(path, block * 2)

    class FakePinned:
        def __init__(self, n: int):
            self.buffer = np.zeros(n, dtype=np.uint8)

        def numpy(self):
            return self.buffer

    reader = BlockStorageReader(path, block)
    try:
        pinned = FakePinned(block)
        read_s, copy_s = reader.read_next_into(pinned)
        assert read_s > 0 and copy_s > 0
        assert int(pinned.buffer.sum(dtype=np.int64)) == expected_range_sum(0, block)
        reader.read_next_into(pinned)
        assert int(pinned.buffer.sum(dtype=np.int64)) == expected_range_sum(block, block)
    finally:
        reader.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'storage_stage'`.

- [ ] **Step 3: Implement storage_stage.py**

Create `experiments/overlap/storage_stage.py`:

```python
"""Storage stage: sequential direct-I/O block reads staged into pinned buffers."""
from __future__ import annotations

import ctypes
import time
from pathlib import Path

import numpy as np

import bootstrap  # noqa: F401  (adds benchmarks/system to sys.path)
from benchmark_storage import DirectReader


class BlockStorageReader:
    """Reads consecutive fixture blocks via FILE_FLAG_NO_BUFFERING.

    Each block lands in the reader's page-aligned kernel buffer and is then
    copied into the caller's pinned tensor. The copy models the RAM staging
    step that a native direct-to-pinned reader would later eliminate; the two
    phases are timed separately so the staging cost is visible in results.
    """

    def __init__(self, path: Path, block_bytes: int):
        self.reader = DirectReader(path.resolve(), block_bytes)
        self.block_bytes = block_bytes
        self.offset = 0

    def read_next_into(self, pinned_tensor) -> tuple[float, float]:
        """Read the next block into ``pinned_tensor``; return (read_s, copy_s)."""
        start = time.perf_counter()
        self.reader.seek(self.offset)
        self.reader.read(self.block_bytes)
        read_seconds = time.perf_counter() - start
        self.offset += self.block_bytes
        source = np.frombuffer(
            (ctypes.c_char * self.block_bytes).from_address(self.reader.buffer),
            dtype=np.uint8,
        )
        start = time.perf_counter()
        pinned_tensor.numpy()[:] = source
        copy_seconds = time.perf_counter() - start
        return read_seconds, copy_seconds

    def reset(self) -> None:
        self.offset = 0

    def close(self) -> None:
        self.reader.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests -v`
Expected: 7 passed.

---

### Task 4: Pipeline context, synchronous variant, and CLI

**Files:**
- Create: `experiments/overlap/variants.py`
- Create: `experiments/overlap/pipeline.py`

- [ ] **Step 1: Implement variants.py with PipelineContext and run_sync**

Create `experiments/overlap/variants.py` (later tasks append the other two runners):

```python
"""Scheduling variants for the layer-streaming overlap experiment."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

import torch


@dataclass
class StageTimes:
    read: list[float] = field(default_factory=list)
    stage_copy: list[float] = field(default_factory=list)
    h2d: list[float] = field(default_factory=list)
    compute: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, list[float]]:
        return {"read": self.read, "stage_copy": self.stage_copy,
                "h2d": self.h2d, "compute": self.compute}


@dataclass
class PipelineContext:
    device: torch.device
    blocks: int
    block_bytes: int
    storage: object                 # BlockStorageReader
    pinned: list                    # ring of 3 pinned uint8 host tensors
    device_buffers: list            # ping-pong pair of uint8 device tensors
    left: torch.Tensor
    right: torch.Tensor
    out: torch.Tensor
    sums: torch.Tensor              # int64 device tensor [blocks]
    copy_stream: torch.cuda.Stream

    def gemm(self) -> None:
        torch.mm(self.left, self.right, out=self.out)


def _timed_pair() -> tuple[torch.cuda.Event, torch.cuda.Event]:
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def run_sync(ctx: PipelineContext) -> StageTimes:
    """Variant 1: fully synchronous read -> stage -> H2D -> compute per block."""
    times = StageTimes()
    device_buffer = ctx.device_buffers[0]
    pinned = ctx.pinned[0]
    for block in range(ctx.blocks):
        read_s, copy_s = ctx.storage.read_next_into(pinned)
        times.read.append(read_s)
        times.stage_copy.append(copy_s)
        h2d_start, h2d_end = _timed_pair()
        h2d_start.record()
        device_buffer.copy_(pinned, non_blocking=False)
        h2d_end.record()
        compute_start, compute_end = _timed_pair()
        compute_start.record()
        ctx.gemm()
        ctx.sums[block] = device_buffer.sum(dtype=torch.int64)
        compute_end.record()
        torch.cuda.synchronize()
        times.h2d.append(h2d_start.elapsed_time(h2d_end) / 1000)
        times.compute.append(compute_start.elapsed_time(compute_end) / 1000)
    return times
```

- [ ] **Step 2: Implement pipeline.py CLI (sync variant wired in)**

Create `experiments/overlap/pipeline.py`:

```python
"""OVR: layer-streaming overlap experiment — storage/H2D/compute scheduling."""
from __future__ import annotations

import argparse
import ctypes
import time
from pathlib import Path

import bootstrap  # noqa: F401  (adds benchmarks/system to sys.path)
from common import RAW_CSV, RESULTS_DIR, rebuild_summary, write_rows

from block_fixture import (DEFAULT_BLOCKS, FP16_LAYER_BYTES, INT4_LAYER_BYTES,
                           expected_range_sum, write_fixture)
from rows import (error_row, makespan_row, memory_row, skipped_row, stage_row,
                  throughput_row, write_timeline)
from storage_stage import BlockStorageReader

SIZES = {"fp16": FP16_LAYER_BYTES, "int4": INT4_LAYER_BYTES}
TIMELINE_CSV = RESULTS_DIR / "overlap_timeline.csv"


class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def process_working_set_bytes() -> int:
    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    if not ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(),
            ctypes.byref(counters), counters.cb):
        raise ctypes.WinError()
    return counters.WorkingSetSize


def verify_checksums(sums_cpu, block_bytes: int, blocks: int) -> list[int]:
    return [b for b in range(blocks)
            if int(sums_cpu[b]) != expected_range_sum(b * block_bytes, block_bytes)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path,
                        default=RESULTS_DIR / "overlap-fixture.bin")
    parser.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--sizes", nargs="+", choices=sorted(SIZES),
                        default=["fp16", "int4"])
    parser.add_argument("--matmul-size", type=int, default=4096)
    parser.add_argument("--keep-file", action="store_true")
    args = parser.parse_args()

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
    runners = {"sync": variants.run_sync}
    rows = []
    try:
        for label in args.sizes:
            block_bytes = SIZES[label]
            storage = BlockStorageReader(args.file, block_bytes)
            try:
                ctx = variants.PipelineContext(
                    device=device,
                    blocks=args.blocks,
                    block_bytes=block_bytes,
                    storage=storage,
                    pinned=[torch.empty(block_bytes, dtype=torch.uint8, pin_memory=True)
                            for _ in range(3)],
                    device_buffers=[torch.empty(block_bytes, dtype=torch.uint8, device=device)
                                    for _ in range(2)],
                    left=torch.randn(args.matmul_size, args.matmul_size,
                                     dtype=torch.float16, device=device),
                    right=torch.randn(args.matmul_size, args.matmul_size,
                                      dtype=torch.float16, device=device),
                    out=torch.empty(args.matmul_size, args.matmul_size,
                                    dtype=torch.float16, device=device),
                    sums=torch.zeros(args.blocks, dtype=torch.int64, device=device),
                    copy_stream=torch.cuda.Stream(),
                )
                ctx.gemm()  # warmup: cuBLAS handles, clock ramp
                ctx.device_buffers[0].copy_(ctx.pinned[0])
                torch.cuda.synchronize()
                for variant, runner in runners.items():
                    for repetition in range(1, args.repetitions + 1):
                        storage.reset()
                        torch.cuda.reset_peak_memory_stats(device)
                        start = time.perf_counter()
                        times = runner(ctx)
                        torch.cuda.synchronize()
                        makespan = time.perf_counter() - start
                        bad = verify_checksums(ctx.sums.cpu(), block_bytes, args.blocks)
                        if bad:
                            rows.append(error_row(
                                f"{variant} checksum", label,
                                f"blocks with wrong data: {bad}"))
                            write_rows(RAW_CSV, rows)
                            rebuild_summary()
                            raise SystemExit(f"checksum mismatch in {variant}/{label}: {bad}")
                        payload = args.blocks * block_bytes
                        rows.append(makespan_row(variant, label, repetition,
                                                 makespan, args.blocks))
                        rows.append(throughput_row(variant, label, repetition,
                                                   payload, makespan))
                        for stage, values in times.as_dict().items():
                            rows.append(stage_row(variant, label, repetition, stage,
                                                  sum(values), args.blocks, block_bytes))
                        rows.append(memory_row(variant, label, "VRAM allocated",
                                               torch.cuda.memory_allocated(device),
                                               "after pass, steady state"))
                        rows.append(memory_row(variant, label, "process working set",
                                               process_working_set_bytes(),
                                               "psapi WorkingSetSize after pass"))
                        write_timeline(TIMELINE_CSV, variant, label, repetition,
                                       times.as_dict())
                        print(f"{label}/{variant} rep {repetition}: "
                              f"makespan {makespan:.3f}s "
                              f"({payload / makespan / 1e9:.3f} GB/s payload)")
            finally:
                storage.close()
    finally:
        if not args.keep_file:
            args.file.unlink(missing_ok=True)
    write_rows(RAW_CSV, rows)
    rebuild_summary()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke-run the sync variant on GPU (small)**

Run: `.venv\Scripts\python.exe experiments\overlap\pipeline.py --sizes fp16 --blocks 4 --repetitions 1 --matmul-size 1024`
Expected: one line `fp16/sync rep 1: makespan ...s (... GB/s payload)`; no checksum error; rows appended to `results/system/raw_measurements.csv`.

- [ ] **Step 4: Verify the recorded rows**

Run: `findstr "^OVR" results\system\raw_measurements.csv | more +0` (or `grep "^OVR" results/system/raw_measurements.csv`)
Expected: rows for `sync makespan`, `sync payload throughput`, `sync stage read/stage_copy/h2d/compute`, `sync VRAM allocated`, `sync process working set`, all `status=ok`.

- [ ] **Step 5: Run full test suite (regression check)**

Run: `.venv\Scripts\python.exe -m pytest tests -v`
Expected: 7 passed (pipeline.py imports no torch at module level, so tests stay GPU-free).

---

### Task 5: Double-buffer variant

**Files:**
- Modify: `experiments/overlap/variants.py` (append `run_double_buffer`)
- Modify: `experiments/overlap/pipeline.py` (register runner)

- [ ] **Step 1: Implement run_double_buffer**

Append to `experiments/overlap/variants.py`:

```python
def run_double_buffer(ctx: PipelineContext) -> StageTimes:
    """Variant 2: storage synchronous; H2D(N+1) on the copy stream overlaps compute(N).

    Buffer safety: the CPU waits for the H2D that last consumed a pinned slot
    before overwriting it; the copy stream waits for the compute that last
    used a device buffer before refilling it.
    """
    times = StageTimes()
    blocks = ctx.blocks
    compute_stream = torch.cuda.current_stream()
    ev_h2d = [torch.cuda.Event() for _ in range(blocks)]
    ev_compute = [torch.cuda.Event() for _ in range(blocks)]
    h2d_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    compute_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

    def enqueue_h2d(block: int) -> None:
        slot = block % 2
        pair = _timed_pair()
        with torch.cuda.stream(ctx.copy_stream):
            pair[0].record()
            ctx.device_buffers[slot].copy_(ctx.pinned[slot], non_blocking=True)
            pair[1].record()
        h2d_pairs.append(pair)
        ev_h2d[block].record(ctx.copy_stream)

    read_s, copy_s = ctx.storage.read_next_into(ctx.pinned[0])
    times.read.append(read_s)
    times.stage_copy.append(copy_s)
    enqueue_h2d(0)

    for block in range(blocks):
        if block + 1 < blocks:
            nxt = (block + 1) % 2
            if block > 0:
                ev_h2d[block - 1].synchronize()                # pinned slot free
                ctx.copy_stream.wait_event(ev_compute[block - 1])  # device slot free
            read_s, copy_s = ctx.storage.read_next_into(ctx.pinned[nxt])
            times.read.append(read_s)
            times.stage_copy.append(copy_s)
            enqueue_h2d(block + 1)
        compute_stream.wait_event(ev_h2d[block])
        pair = _timed_pair()
        pair[0].record(compute_stream)
        ctx.gemm()
        ctx.sums[block] = ctx.device_buffers[block % 2].sum(dtype=torch.int64)
        pair[1].record(compute_stream)
        compute_pairs.append(pair)
        ev_compute[block].record(compute_stream)

    torch.cuda.synchronize()
    times.h2d = [start.elapsed_time(end) / 1000 for start, end in h2d_pairs]
    times.compute = [start.elapsed_time(end) / 1000 for start, end in compute_pairs]
    return times
```

- [ ] **Step 2: Register the runner**

In `experiments/overlap/pipeline.py`, change:

```python
    runners = {"sync": variants.run_sync}
```

to:

```python
    runners = {"sync": variants.run_sync, "double_buffer": variants.run_double_buffer}
```

- [ ] **Step 3: Smoke-run on GPU (small)**

Run: `.venv\Scripts\python.exe experiments\overlap\pipeline.py --sizes fp16 --blocks 4 --repetitions 1 --matmul-size 1024`
Expected: lines for both `fp16/sync` and `fp16/double_buffer`; no checksum mismatch (a race in buffer reuse would surface here as a checksum error).

- [ ] **Step 4: Eyeball stage sums**

Run: `grep "double_buffer" results/system/raw_measurements.csv | tail -7`
Expected: stage rows exist; `stage h2d` and `stage compute` sums are each far below the sync makespan (they now overlap storage), while makespan is below or near sync's.

---

### Task 6: Three-stage variant

**Files:**
- Modify: `experiments/overlap/variants.py` (append `run_three_stage`)
- Modify: `experiments/overlap/pipeline.py` (register runner)

- [ ] **Step 1: Implement run_three_stage**

Append to `experiments/overlap/variants.py`:

```python
def run_three_stage(ctx: PipelineContext) -> StageTimes:
    """Variant 3: storage(N+2) in a worker thread, H2D(N+1) on the copy stream,
    compute(N) on the default stream.

    Buffer safety: the worker overwrites pinned slot b%3 only after the H2D of
    block b-3 completed; ``h2d_enqueued`` closes the race where the worker
    could reach that wait before the main thread has recorded the event.
    """
    times = StageTimes(read=[0.0] * ctx.blocks, stage_copy=[0.0] * ctx.blocks)
    blocks = ctx.blocks
    compute_stream = torch.cuda.current_stream()
    storage_done = [threading.Event() for _ in range(blocks)]
    h2d_enqueued = [threading.Event() for _ in range(blocks)]
    ev_h2d = [torch.cuda.Event() for _ in range(blocks)]
    ev_compute = [torch.cuda.Event() for _ in range(blocks)]
    h2d_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    compute_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

    def storage_worker() -> None:
        for block in range(blocks):
            slot = block % 3
            if block >= 3:
                h2d_enqueued[block - 3].wait()
                ev_h2d[block - 3].synchronize()  # pinned slot free
            read_s, copy_s = ctx.storage.read_next_into(ctx.pinned[slot])
            times.read[block] = read_s
            times.stage_copy[block] = copy_s
            storage_done[block].set()

    def enqueue_h2d(block: int) -> None:
        pair = _timed_pair()
        with torch.cuda.stream(ctx.copy_stream):
            pair[0].record()
            ctx.device_buffers[block % 2].copy_(ctx.pinned[block % 3], non_blocking=True)
            pair[1].record()
        h2d_pairs.append(pair)
        ev_h2d[block].record(ctx.copy_stream)
        h2d_enqueued[block].set()

    worker = threading.Thread(target=storage_worker, name="storage-stage")
    worker.start()
    try:
        storage_done[0].wait()
        enqueue_h2d(0)
        for block in range(blocks):
            if block + 1 < blocks:
                storage_done[block + 1].wait()
                if block > 0:
                    ctx.copy_stream.wait_event(ev_compute[block - 1])  # device slot free
                enqueue_h2d(block + 1)
            compute_stream.wait_event(ev_h2d[block])
            pair = _timed_pair()
            pair[0].record(compute_stream)
            ctx.gemm()
            ctx.sums[block] = ctx.device_buffers[block % 2].sum(dtype=torch.int64)
            pair[1].record(compute_stream)
            compute_pairs.append(pair)
            ev_compute[block].record(compute_stream)
    finally:
        worker.join()

    torch.cuda.synchronize()
    times.h2d = [start.elapsed_time(end) / 1000 for start, end in h2d_pairs]
    times.compute = [start.elapsed_time(end) / 1000 for start, end in compute_pairs]
    return times
```

- [ ] **Step 2: Register the runner**

In `experiments/overlap/pipeline.py`, change the runners dict to:

```python
    runners = {
        "sync": variants.run_sync,
        "double_buffer": variants.run_double_buffer,
        "three_stage": variants.run_three_stage,
    }
```

- [ ] **Step 3: Smoke-run on GPU (small)**

Run: `.venv\Scripts\python.exe experiments\overlap\pipeline.py --sizes fp16 --blocks 6 --repetitions 1 --matmul-size 1024`
Expected: lines for all three variants; no checksum mismatch (6 blocks exercises the `block >= 3` ring-recycling path); process exits 0.

---

### Task 7: Canonical full run and result inspection

**Files:** none modified (measurement task)

- [ ] **Step 1: Full canonical run**

Close storage-heavy apps; keep machine on AC power. Run:

```
.venv\Scripts\python.exe experiments\overlap\pipeline.py --sizes fp16 int4 --blocks 36 --repetitions 3
```

Expected: 18 lines (2 sizes × 3 variants × 3 reps). Rough duration: fixture write ~14 GB once, then ~60 s of FP16 passes and ~20 s of INT4 passes. Fixture deleted at the end.

- [ ] **Step 2: Inspect the summary**

Run: `grep "^OVR" results/system_characterization.csv`
Expected: median rows per variant for makespan, throughput, and each stage, for both block sizes.

- [ ] **Step 3: Check the hypotheses against the numbers**

Compute and note (do not hardcode into code — this is analysis):
- Does `sync makespan ≈ stage read + stage_copy + h2d + compute` (within a few %)?
- Are `double_buffer`/`three_stage` makespans close to `stage read + stage_copy` (i.e., H2D+compute hidden)?
- FP16: read stage ≈ 153 ms/block expected from B01 (0.387 GB / 2.526 GB/s). INT4: ≈ 47 ms/block.
- stage_copy ≈ 22 ms/block FP16 (0.387 GB / 17.8 GB/s B03): is it a visible critical-path cost?
Record observations verbatim for the docs task; if a hypothesis fails, that is a result — record it, do not tune it away.

---

### Task 8: Documentation

**Files:**
- Create: `experiments/overlap/README.md`
- Modify: `AGENTS.md` (layout + run commands)
- Modify: `docs/system-characterization.md` (follow-up experiment status)

- [ ] **Step 1: Write experiments/overlap/README.md**

Content requirements (write it measured-tone, English):
- Purpose: the overlap gate for the provisional path-C recommendation; maps to the NVMe → RAM → pinned → VRAM pipeline.
- The three variants and their buffer-reuse safety rules (pinned slot waits for its last H2D; device slot waits for its last compute).
- The staging copy (aligned kernel buffer → pinned) is explicit and separately timed; a native direct-to-pinned reader is a possible later native component **if** the measured stage_copy cost justifies it.
- Checksum design: cell pattern, analytic expected sums, async GPU verification; limitation — constant-per-cell bytes catch wrong-block/torn-buffer bugs, not every conceivable corruption.
- Confounders: WDDM scheduling, desktop GPU contention, single-reader queue depth 1, fixture deleted by default (`--keep-file` to retain), first pass after fixture creation may see different storage behavior.
- Run commands (smoke and canonical) exactly as in Tasks 4–7; outputs: raw rows, summary rows, `results/system/overlap_timeline.csv`.

- [ ] **Step 2: Update AGENTS.md**

- Repository layout block: add `experiments/overlap/` entry (no longer an empty placeholder) and `tests/test_overlap_logic.py`.
- Build and run commands: add the canonical OVR command.
- Testing section: replace "There is no test suite yet" with: pytest suite in `tests/` covering experiment logic (run `.venv\Scripts\python.exe -m pytest tests -v`); benchmarks remain the verification mechanism for hardware claims.

- [ ] **Step 3: Update docs/system-characterization.md**

In "Smallest useful follow-up experiment", mark the experiment as implemented in `experiments/overlap/` and summarize the measured outcome from Task 7 Step 3 in two or three sentences (makespan vs stage-sum comparison per variant, whether overlap was achieved, staging-copy cost). Keep hypotheses clearly separated from measured facts. Update the "Current limitations" line "the overlap question is untouched by everything measured so far" to reflect reality after the run.

- [ ] **Step 4: Final regression**

Run: `.venv\Scripts\python.exe -m pytest tests -v`
Expected: 7 passed.

---

## Self-review notes (completed by the author)

- Spec coverage: three variants ✓ (Tasks 4–6); 36 × FP16-layer blocks + INT4 variant ✓ (constants, Task 7); checksum correctness ✓ (fixture module + `verify_checksums`); per-stage + end-to-end + bytes + working set + VRAM recording ✓ (Task 4 pipeline); timeline artifact ✓ (`write_timeline`); reuse of common.py result machinery ✓.
- The spec's "profiler timeline" is satisfied by the per-block per-stage timeline CSV; a `torch.profiler` chrome trace was considered and dropped (YAGNI — durations CSV answers the overlap question; adding it later is one flag).
- Name consistency checked: `StageTimes.as_dict` keys (`read/stage_copy/h2d/compute`) match row stage names; `BlockStorageReader.read_next_into/reset/close` used identically in all variants and tests; `enqueue_h2d` semantics differ per variant (slot arithmetic) and are defined locally inside each runner.
