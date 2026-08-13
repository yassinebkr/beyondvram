"""Pure validation tests for the real-layer package manifest."""
from __future__ import annotations

import json

import pytest

from qwen_layer_stream import ALIGNMENT, align_up, load_manifest


def test_align_up_preserves_and_advances_alignment() -> None:
    assert align_up(0) == 0
    assert align_up(ALIGNMENT) == ALIGNMENT
    assert align_up(ALIGNMENT + 1) == 2 * ALIGNMENT


def test_load_manifest_rejects_unaligned_tensor_offset(tmp_path) -> None:
    manifest = {
        "format": "beyondvram-real-layer-v1",
        "alignment_bytes": ALIGNMENT,
        "tensors": [{"name": "x", "shape": [1], "offset": 1, "nbytes": 2}],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="unaligned"):
        load_manifest(path)
