"""Pure-logic tests for the overlap pipeline experiment (no GPU required)."""
from __future__ import annotations

import os

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
    for offset_cells, size_cells in [(0, 600), (3, 251), (100, 77), (200, 100)]:
        offset, size = offset_cells * CELL_BYTES, size_cells * CELL_BYTES
        assert expected_range_sum(offset, size) == int(raw[offset:offset + size].sum(dtype=np.int64))


def test_expected_range_sum_rejects_misalignment():
    with pytest.raises(ValueError):
        expected_range_sum(1, CELL_BYTES)
    with pytest.raises(ValueError):
        expected_range_sum(0, CELL_BYTES - 1)


def test_rows_use_known_fields_and_units():
    from rows import makespan_row, memory_row, stage_row, throughput_row
    from common import FIELDS  # importable once rows' bootstrap side effect ran

    stage = stage_row("sync", "fp16", 1, "read", 1.5, 36, FP16_LAYER_BYTES)
    assert set(stage) <= set(FIELDS)
    assert stage["benchmark_id"] == "OVR"
    assert stage["unit"] == "s" and stage["status"] == "ok"
    assert makespan_row("sync", "fp16", 1, 7.0, 36)["value"] == 7.0
    throughput = throughput_row("sync", "fp16", 1, 36 * FP16_LAYER_BYTES, 7.0)
    assert throughput["unit"] == "GB/s_payload"
    assert throughput["value"] == pytest.approx(36 * FP16_LAYER_BYTES / 7.0 / 1e9)
    assert memory_row("sync", "fp16", 1, "process working set", 1234, "x")["unit"] == "bytes"


def test_rows_reject_invalid_measurements(tmp_path):
    from rows import stage_row, throughput_row, write_timeline

    with pytest.raises(ValueError):
        throughput_row("sync", "fp16", 1, 1, 0.0)
    with pytest.raises(ValueError):
        stage_row("sync", "fp16", 1, "unknown", 1.0, 1, CELL_BYTES)
    with pytest.raises(ValueError):
        write_timeline(tmp_path / "timeline.csv", "sync", "fp16", 1,
                       {"read": [0.1], "stage_copy": [], "h2d": [0.1], "compute": [0.1]})


def test_checksum_verification_identifies_only_bad_blocks():
    from pipeline import verify_checksums

    block = CELL_BYTES * FILL_PERIOD
    expected = [expected_range_sum(index * block, block) for index in range(3)]
    observed = [expected[0], expected[1] + 1, expected[2]]
    assert verify_checksums(observed, block, 3) == [1]


def test_timeline_writer_appends_rows(tmp_path):
    from rows import write_timeline

    target = tmp_path / "timeline.csv"
    write_timeline(target, "sync", "fp16", 1, {
        "read": [0.1, 0.2], "stage_copy": [0.01, 0.01],
        "h2d": [0.01, 0.01], "compute": [0.02, 0.02],
    })
    write_timeline(target, "sync", "fp16", 2, {
        "read": [0.3], "stage_copy": [0.01], "h2d": [0.01], "compute": [0.02],
    })
    lines = target.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "variant,block_label,repetition,block,stage,seconds"
    assert len(lines) == 1 + 8 + 4


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
