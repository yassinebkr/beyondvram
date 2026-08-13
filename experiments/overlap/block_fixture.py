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
