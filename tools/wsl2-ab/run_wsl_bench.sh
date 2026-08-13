#!/bin/bash
# BeyondVRAM experiment 2 (docs/next-experiments.md): WSL2 OS-tax A/B.
# Runs inside the BeyondVRAM-Test WSL2 distro (ubuntu-base 24.04.4, root).
# Protocol mirrors the native placement grid exactly: b10355 llama-bench,
# 128 prompt tokens, 32 generated, 3 repetitions, mmap, f16 KV, default threads.
# Output lands in the repo results dir via /mnt/c.
set -euo pipefail

B=/opt/llama-b10355/llama-bench
M=/root/model.gguf
OUT=/mnt/c/Users/yassi/Documents/code/BeyondVram/results/gpt-oss

# Environment record (kernel, memory, vmmem view) before any benchmark.
{
  echo "== uname =="; uname -a
  echo "== free =="; free -m
  echo "== cpuinfo model =="; grep -m1 "model name" /proc/cpuinfo
  echo "== date =="; date -u +%FT%TZ
} > "$OUT/wsl-ab-env.txt"

run() { # name, ngl, n-cpu-moe
  echo "[wsl-ab] $1 ngl=$2 n-cpu-moe=$3"
  "$B" -m "$M" -ngl "$2" --n-cpu-moe "$3" -p 128 -n 32 -r 3 -o json \
    > "$OUT/wsl-ab-$1.json"
}

run cpu00 0 0      # pure CPU floor (native reference: 15.57 tok/s)
# CUDA configs (moe24, moe10) run after the in-VM CUDA build exists:
# run moe24 24 24  # all-experts-CPU (native reference: 21.00 / 19.74 tok/s)
# run moe10 24 10  # placement optimum (native reference: 44.79 / 41.78 tok/s)

echo "[wsl-ab] done"
