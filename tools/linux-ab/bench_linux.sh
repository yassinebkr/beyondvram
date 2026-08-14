#!/bin/bash
# BeyondVRAM experiment 6 (docs/next-experiments.md): bare-metal Linux
# OS-tax A/B. Runs on the minimal Debian dual-boot (see SETUP.md).
#
# Mirrors the native/WSL2 protocol exactly: b10355 llama-bench,
# 128 prompt tokens, 32 generated, mmap, f16 KV, default threads — but
# 5 reps, because arms cannot interleave across reboots and drift control
# comes from alternating boot cycles instead (>=3 per OS).
#
# Configs reproduce the experiment-2 arms on gpt-oss-20b so the result is
# a 3-way table (native Windows / WSL2 / bare metal), plus the Qwen3-30B
# Track-1/#4 arms (stock and the mid24-Q3_K variant) when the files exist.
# Missing model files are skipped, never fabricated.
#
# Usage: ./bench_linux.sh [out_dir]     (default: ./linux-ab-results)
set -uo pipefail

BIN=${BIN:-/opt/llama-b10355}
BENCH="$BIN/llama-bench"
OUT=${1:-./linux-ab-results}
mkdir -p "$OUT"

{ # environment record before any benchmark
  echo "== uname =="; uname -a
  echo "== free =="; free -m
  echo "== cpuinfo model =="; grep -m1 "model name" /proc/cpuinfo
  echo "== governor =="; cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "n/a"
  echo "== ram =="; sudo -n dmidecode -t memory 2>/dev/null | grep -E "Speed:|Configured" | sort -u || echo "dmidecode needs root"
  echo "== gpu =="; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"
  echo "== date =="; date -u +%FT%TZ
} | tee "$OUT/env.txt"

step=0
run() { # name, model, ngl, n-cpu-moe
  local name=$1 model=$2 ngl=$3 ncpu=$4
  step=$((step + 1))
  if [ ! -f "$model" ]; then
    echo "[$(date +%H:%M:%S)] [step $step] $name: SKIPPED (model not present: $model)"
    return
  fi
  echo "[$(date +%H:%M:%S)] [step $step] $name: ngl=$ngl n-cpu-moe=$ncpu model=$(basename "$model")"
  "$BENCH" -m "$model" -ngl "$ngl" --n-cpu-moe "$ncpu" \
    -p 128 -n 32 -r 5 -o json > "$OUT/$name.json" \
    || echo "[$(date +%H:%M:%S)] [step $step] $name: FAILED rc=$? (row kept as empty json)"
}

GPTOSS=${GPTOSS:-/root/models/gpt-oss-20b-MXFP4.gguf}
QWEN_STOCK=${QWEN_STOCK:-/root/models/Qwen3-30B-A3B-Q4_K_M.gguf}
QWEN_MID24=${QWEN_MID24:-/root/models/qwen3-30b-a3b-mid24-q3k.gguf}

# gpt-oss-20b arms — the experiment-2 reference configs
run gptoss-cpu00   "$GPTOSS" 0  0
run gptoss-moe24   "$GPTOSS" 24 24
run gptoss-moe10   "$GPTOSS" 24 10
# Qwen3-30B arms — Track-1 optimum and the #4 variant optimum
run qwen3-stock    "$QWEN_STOCK" 48 33
run qwen3-mid24q3k "$QWEN_MID24" 48 30

echo "[$(date +%H:%M:%S)] done -> $OUT"
