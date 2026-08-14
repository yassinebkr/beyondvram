"""Collect software and hardware metadata without optional dependencies.

The nvidia-smi/nvcc probes run for seconds, so their output stays captured
rather than streamed. Ctrl+C aborts without writing output, exit code 130.
"""

from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from common import RESULTS_DIR, dump_json, ts, utc_now


class MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_ulong),
        ("memory_load", ctypes.c_ulong),
        ("total_phys", ctypes.c_ulonglong),
        ("avail_phys", ctypes.c_ulonglong),
        ("total_page_file", ctypes.c_ulonglong),
        ("avail_page_file", ctypes.c_ulonglong),
        ("total_virtual", ctypes.c_ulonglong),
        ("avail_virtual", ctypes.c_ulonglong),
        ("avail_extended_virtual", ctypes.c_ulonglong),
    ]


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def main() -> None:
    print(f"[{ts()}] collecting system info: memory, disk, nvidia-smi, nvcc, "
          f"package versions", flush=True)
    print(f"[{ts()}] output -> {RESULTS_DIR / 'system-info.json'} (Ctrl+C aborts; "
          f"nothing partial is written)", flush=True)
    try:
        memory = MemoryStatus()
        memory.length = ctypes.sizeof(memory)
        memory_ok = bool(ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory))) if os.name == "nt" else False
        root_usage = shutil.disk_usage(Path.cwd().anchor)
        gpu_query = command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.free,pci.bus_id,compute_cap",
                "--format=csv,noheader,nounits",
            ]
        )
        cpu_name = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown")
        payload = {
            "captured_at_utc": utc_now(),
            "os": platform.platform(),
            "python": {"version": sys.version, "executable": sys.executable},
            "cpu": {"name": cpu_name, "logical_processors": os.cpu_count()},
            "ram": {
                "total_bytes": memory.total_phys if memory_ok else None,
                "available_bytes": memory.avail_phys if memory_ok else None,
            },
            "workspace_volume": {
                "root": Path.cwd().anchor,
                "total_bytes": root_usage.total,
                "free_bytes": root_usage.free,
            },
            "gpu_nvidia_smi_csv": gpu_query,
            "nvidia_smi": command_output(["nvidia-smi"]),
            "cuda_toolkit_nvcc": command_output(["nvcc", "--version"]),
            "python_packages": {
                "numpy": command_output([sys.executable, "-c", "import numpy; print(numpy.__version__)"]),
                "torch": command_output([sys.executable, "-c", "import torch; print(torch.__version__); print(torch.version.cuda)"]),
            },
        }
        dump_json(RESULTS_DIR / "system-info.json", payload)
    except KeyboardInterrupt:
        print(f"\n[{ts()}] interrupted — no output written", flush=True)
        sys.exit(130)
    print(f"[{ts()}] wrote {RESULTS_DIR / 'system-info.json'}", flush=True)


if __name__ == "__main__":
    main()

