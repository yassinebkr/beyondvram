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
