# System characterization benchmarks

These scripts characterize the actual Windows host before an inference architecture is frozen.

## Quick run

From the repository root:

```powershell
python benchmarks/system/run_all.py
```

The default storage run creates a temporary 2 GiB fixture on the workspace volume, performs five direct-I/O repetitions, and removes the fixture. Use `benchmark_storage.py --help` to select another volume or workload. Close storage-heavy applications and keep the machine on AC power. Do not compare runs made under different GPU power policies without recording that fact.

`benchmark_cuda.py` requires a CUDA-enabled PyTorch installation. When unavailable it records B04–B06 as skipped, rather than inventing results. The project benchmark environment is the local `.venv` (Python 3.14, torch 2.13.0+cu126); recreate it from the repository root with:

```powershell
uv venv .venv --python 3.14
uv pip install --python .venv/Scripts/python.exe "torch==2.13.0+cu126" --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv/Scripts/python.exe numpy pillow
```

Note that the PyPI Windows wheel for torch is CPU-only; the CUDA build comes from the PyTorch index above. Run the CUDA and plotting scripts with `.venv\Scripts\python.exe` (or launch `run_all.py` with it). The RTX 3070 Ti is compute capability 8.6, which the cu126 build supports.

Outputs:

- `results/system/system-info.json`: environment snapshot
- `results/system/raw_measurements.csv`: every repetition
- `results/system_characterization.csv`: medians, ranges, and standard deviations
- `plots/system_memory_hierarchy.png`: measured bandwidths only

Interpret `GB/s_payload` as bytes useful to the operation divided by elapsed time. For RAM copy, physical memory traffic is approximately twice the payload (one read plus one write). The storage script uses Windows `FILE_FLAG_NO_BUFFERING`; sizes and offsets are 4096-byte aligned to avoid measuring the filesystem cache.

