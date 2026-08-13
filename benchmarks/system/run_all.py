"""Run all characterization scripts; each benchmark owns its raw records."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def run(script: str, *arguments: str) -> None:
    command = [sys.executable, str(HERE / script), *arguments]
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    run("collect_system_info.py")
    run("benchmark_storage.py")
    run("benchmark_ram.py")
    run("benchmark_ram_read.py")
    run("benchmark_cuda.py")
    run("plot_results.py")


if __name__ == "__main__":
    main()

