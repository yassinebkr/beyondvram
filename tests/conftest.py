"""Pytest path setup for the experiment modules."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for candidate in (ROOT, ROOT / "experiments" / "overlap", ROOT / "experiments" / "real_layer"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
