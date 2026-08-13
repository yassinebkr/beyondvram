#!/bin/bash
# BeyondVRAM experiment 2b (docs/next-experiments.md): paired hybrid A/B,
# native Windows CI binary (MSVC, win-cuda-13.3) vs in-VM gcc CUDA build,
# both pinned at b10355 (dd1ea52). Configs: (24,24) all-experts-CPU and the
# (24,10) placement optimum. Paired alternating arms, same drift controls as
# ab_paired.sh: VM stopped before native arms, model page cache warmed.
#
# Runs from Git Bash at the repo root. ~20 minutes. Outputs: results/gpt-oss/ab3-*.
set -euo pipefail
cd "$(dirname "$0")/../.."

NB="tools/llama.cpp-b10355/llama-bench.exe"
M="models/gpt-oss-20b-GGUF/gpt-oss-20b-MXFP4.gguf"
R="results/gpt-oss"
WB="/root/llama.cpp/build/bin/llama-bench"   # in-VM gcc CUDA build (b10355)
WM="/root/model.gguf"

arm() { # $1 = side (native|wsl), $2 = config tag, $3 = n-cpu-moe
  local side="$1" tag="$2" k="$3"
  if [ "$side" = native ]; then
    cat "$M" > /dev/null
    "$NB" -m "$M" -ngl 24 --n-cpu-moe "$k" -p 128 -n 32 -r 3 -o json \
      > "$R/ab3-native-$tag.json" 2> "$R/ab3-native-$tag.stderr.txt"
  else
    MSYS2_ARG_CONV_EXCL="*" wsl -d BeyondVRAM-Test -u root -- bash -c \
      "cat $WM > /dev/null && $WB -m $WM -ngl 24 --n-cpu-moe $k -p 128 -n 32 -r 3 -o json" \
      > "$R/ab3-wsl-$tag.json"
  fi
  echo "[ab3] $side $tag done" >&2
}

for round in a b; do
  echo "[ab3] round $round moe24" >&2
  wsl --shutdown; sleep 5
  arm native "moe24-$round" 24
  arm wsl    "moe24-$round" 24
  echo "[ab3] round $round moe10" >&2
  wsl --shutdown; sleep 5
  arm native "moe10-$round" 10
  arm wsl    "moe10-$round" 10
done

wsl --shutdown || true
echo "[ab3] all done" >&2
