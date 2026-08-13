#!/bin/bash
# BeyondVRAM experiment 2 (docs/next-experiments.md): paired native-vs-WSL2 A/B.
#
# Why paired: native pure-CPU llama-bench drifted ~15% between sessions on the
# same config (15.57 grid-era vs 13.15 recheck), which invalidates
# cross-session A/B comparisons. This orchestrator alternates arms in one
# session, stopping the WSL VM before each native arm (a resident VM holds RAM
# and perturbs the native side) and warming the model into page cache before
# every arm (cold-page first-rep cost otherwise lands inside avg_ts).
#
# Runs from Git Bash at the repo root. ~20 minutes. Outputs: results/gpt-oss/ab2-*.
set -euo pipefail
cd "$(dirname "$0")/../.."

NB="tools/llama.cpp-b10355/llama-bench.exe"
M="models/gpt-oss-20b-GGUF/gpt-oss-20b-MXFP4.gguf"
R="results/gpt-oss"
WB="/opt/llama-b10355/llama-bench"   # WSL-side official CPU release
WM="/root/model.gguf"

native_arm() { # $1 = tag
  cat "$M" > /dev/null
  "$NB" -m "$M" -ngl 0 -p 128 -n 32 -r 3 -o json \
    > "$R/ab2-native-cpu00-$1.json" 2> "$R/ab2-native-cpu00-$1.stderr.txt"
  echo "[ab2] native $1 done" >&2
}

wsl_arm() { # $1 = tag, rest = extra llama-bench args
  local tag="$1"; shift
  MSYS2_ARG_CONV_EXCL="*" wsl -d BeyondVRAM-Test -u root -- bash -c \
    "cat $WM > /dev/null && $WB -m $WM -ngl 0 $* -p 128 -n 32 -r 3 -o json" \
    > "$R/ab2-wsl-cpu00-$tag.json"
  echo "[ab2] wsl $tag done" >&2
}

echo "[ab2] round A" >&2
wsl --shutdown; sleep 5
native_arm a
wsl_arm a            # default threads (6)
wsl_arm t6  -t 6     # physical cores only
wsl_arm t12 -t 12    # all SMT threads

# zen4 forced-variant load test: if the zen4 backend cannot load on Zen 3,
# haswell was the correct pick on both OSes and no dispatch artifact exists.
MSYS2_ARG_CONV_EXCL="*" wsl -d BeyondVRAM-Test -u root -- bash -c "
mkdir -p /opt/llama-zen4 && cp /opt/llama-b10355/*.so /opt/llama-zen4/ && cp /opt/llama-b10355/llama-bench /opt/llama-zen4/
cd /opt/llama-zen4 && rm -f libggml-cpu-{haswell,cascadelake,cooperlake,icelake,alderlake,cannonlake,sapphirerapids,skylakex,ivybridge,sandybridge,piledriver}.so
ls libggml-cpu*.so
./llama-bench -m $WM -ngl 0 -p 16 -n 1 -r 1 2>&1 | grep -iE 'backend|zen|haswell|error|illegal' | head -5
" > "$R/ab2-wsl-zen4-forced.txt" 2>&1 || true
echo "[ab2] zen4 forced test done" >&2

echo "[ab2] round B" >&2
wsl --shutdown; sleep 5
native_arm b
wsl_arm b

echo "[ab2] round C" >&2
wsl --shutdown; sleep 5
native_arm c

wsl --shutdown || true
echo "[ab2] all done" >&2
