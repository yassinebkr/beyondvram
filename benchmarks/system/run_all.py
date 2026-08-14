"""Run all characterization scripts; each benchmark owns its raw records.

Children inherit the console, so their live output streams directly. Ctrl+C
stops the suite with exit code 130; completed steps keep their results.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from common import ts


HERE = Path(__file__).resolve().parent

STEPS = [
    "collect_system_info.py",
    "benchmark_storage.py",
    "benchmark_ram.py",
    "benchmark_ram_read.py",
    "benchmark_cuda.py",
    "plot_results.py",
]


def run(script: str, *arguments: str) -> None:
    command = [sys.executable, str(HERE / script), *arguments]
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    print(f"[{ts()}] run_all: {len(STEPS)} characterization steps; results land in "
          "results/system/ and results/system_characterization.csv", flush=True)
    print(f"[{ts()}] Ctrl+C stops the suite; each child persists its own partial "
          "rows", flush=True)
    try:
        for index, script in enumerate(STEPS, 1):
            print(f"[{ts()}] [step {index}/{len(STEPS)}] {script}", flush=True)
            run(script)
    except KeyboardInterrupt:
        print(f"\n[{ts()}] interrupted — suite stopped; completed steps keep "
              "their results", flush=True)
        sys.exit(130)
    print(f"[{ts()}] run_all complete", flush=True)


if __name__ == "__main__":
    main()

