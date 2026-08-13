#!/bin/bash
# BeyondVRAM experiment 2b: in-VM CUDA toolchain + pinned llama.cpp b10355 build.
# Runs as root inside the BeyondVRAM-Test WSL2 distro. Idempotent.
# sm_86-only arch list: the RTX 3070 Ti is the only target, and the native CI
# binary dispatches the same sm_86 kernels.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

apt-get install -y -qq wget git build-essential cmake >/dev/null

if ! command -v nvcc >/dev/null 2>&1; then
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
  dpkg -i /tmp/cuda-keyring.deb >/dev/null
  apt-get update -qq
  apt-get install -y -qq cuda-toolkit-13-3 >/dev/null 2>&1 || apt-get install -y -qq cuda-toolkit-13 >/dev/null
fi
export PATH=/usr/local/cuda/bin:$PATH
nvcc --version | tail -1

if [ ! -d /root/llama.cpp ]; then
  git clone --depth 1 -b b10355 https://github.com/ggml-org/llama.cpp /root/llama.cpp
fi
cd /root/llama.cpp
git log --oneline -1
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 2>&1 | tail -3
cmake --build build --target llama-bench -j "$(nproc)" 2>&1 | tail -3
ls -la build/bin/llama-bench
