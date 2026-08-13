"""Add benchmarks/system to sys.path so experiments reuse the existing helpers."""
from __future__ import annotations

import sys
from pathlib import Path

SYSTEM_BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks" / "system"
if str(SYSTEM_BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(SYSTEM_BENCHMARKS))
