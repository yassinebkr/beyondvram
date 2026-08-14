#!/bin/bash
# bench-tui validation battery (spec: docs/superpowers/specs/2026-08-14-bench-tui-design.md).
# Builds, runs unit tests, pty-rendering smoke, and the measurement-contamination
# A/B (same tiny CPU-only llama-bench config with the TUI idle-running vs absent).
# Nothing here touches a real measured run.
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
BUILD=${BUILD:-$HERE/build}
LLAMA=${LLAMA:-$HOME/llama-b10355}
V=${V:-$HOME/bench-tui-validation}
step=0
banner() { step=$((step + 1)); echo "[$(date +%H:%M:%S)] [step $step] $*"; }
fail() { echo "FAIL: $*"; exit 1; }
mkdir -p "$V" || fail "cannot create $V"

banner "build"
cmake -S "$HERE" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release || fail configure
cmake --build "$BUILD" -j"$(nproc)" || fail build

banner "unit tests"
"$BUILD/bench-tui-tests" || fail tests

banner "pty smoke: tabs render, q quits"
{ sleep 2; printf "32"; sleep 2; printf q; sleep 1; } | \
  TERM=xterm-256color timeout 15 script -qefc \
  "stty rows 40 cols 110; $BUILD/bench-tui --config-dir $V/cfg --out-dir $V/runs --bin-dir $LLAMA --models-dir $HOME/models" \
  "$V/smoke.typescript" >/dev/null \
  || fail "pty smoke pipeline: crash after render, ignored q (timeout 124), or writer failure"
grep -q Presets "$V/smoke.typescript" || fail "runs tab missing"
grep -q "RAM" "$V/smoke.typescript" || fail "system tab missing"
grep -q Agents "$V/smoke.typescript" || fail "agents tab missing"
echo "SMOKE-OK"

banner "llama b10355 binaries"
if [ ! -x "$LLAMA/llama-bench" ]; then
  mkdir -p "$LLAMA"
  tar -xzf "$REPO/tools/wsl2-ab/llama-b10355-bin-ubuntu-x64.tar.gz" -C "$LLAMA" \
    || fail "tarball missing: tools/wsl2-ab/llama-b10355-bin-ubuntu-x64.tar.gz"
  inner=$(find "$LLAMA" -name llama-bench -type f | head -1)
  [ -n "$inner" ] || fail "llama-bench not in tarball"
  dir=$(dirname "$inner")
  [ "$dir" = "$LLAMA" ] || mv "$dir"/* "$LLAMA/" || fail "tarball layout flatten failed"
  chmod +x "$LLAMA"/llama-* || fail "chmod llama binaries"
fi
"$LLAMA/llama-bench" --version | head -1

banner "small local model"
mkdir -p "$HOME/models"
if ! ls "$HOME/models"/*.gguf >/dev/null 2>&1; then
  src=$(ls "$REPO"/models/Qwen3-0.6B-GGUF/*.gguf 2>/dev/null | head -1)
  [ -n "$src" ] || fail "no GGUF under models/Qwen3-0.6B-GGUF"
  cp "$src" "$HOME/models/" || fail "model copy to $HOME/models"
fi
MODEL=$(ls "$HOME/models"/*.gguf | head -1)
[ -n "$MODEL" ] || fail "no model after copy"
echo "model: $MODEL"

banner "ready-made smoke preset + shell-only agents for the operator live test"
mkdir -p "$V/cfg"
cat > "$V/cfg/presets.json" <<EOF
[{"name":"smoke-06b-cpu","model_path":"$MODEL","binary":"llama-bench","args":"-m $MODEL -ngl 0 -p 128 -n 32 -r 3 -o json","repeats":1}]
EOF
cat > "$V/cfg/agents.json" <<EOF
[{"name":"shell","command":"\${SHELL:-/bin/bash}","workdir":"$HOME"}]
EOF

banner "contamination A/B: 3 reps bare, 3 reps with TUI idle"
run_rep() {
  "$LLAMA/llama-bench" -m "$MODEL" -ngl 0 -p 128 -n 32 -r 3 -o json \
    > "$V/$1-$2.json" 2> "$V/$1-$2.log" || fail "llama-bench $1-$2"
}
for i in 1 2 3; do run_rep bare "$i"; done
{ sleep 3600 & echo $! >"$V/.idle-writer.pid"; wait; } | \
  TERM=xterm-256color script -qefc \
  "$BUILD/bench-tui --config-dir $V/cfg --out-dir $V/runs --bin-dir $LLAMA --models-dir $HOME/models" \
  "$V/idle.typescript" >/dev/null &
TUI_PID=$!
sleep 2
kill -0 "$TUI_PID" 2>/dev/null || fail "idle TUI not running"
for i in 1 2 3; do run_rep tui "$i"; done
kill -0 "$TUI_PID" 2>/dev/null || fail "idle TUI died during the tui reps"
kill "$TUI_PID" 2>/dev/null
kill "$(cat "$V/.idle-writer.pid")" 2>/dev/null
wait "$TUI_PID" 2>/dev/null
rm -f "$V/.idle-writer.pid"

tg_of() { grep -o '"avg_ts": *[0-9.]*' "$1" | tail -1 | grep -o '[0-9.]*'; }
med3() { printf '%s\n' "$1" "$2" "$3" | sort -n | sed -n 2p; }
for f in bare-1 bare-2 bare-3 tui-1 tui-2 tui-3; do
  [ -n "$(tg_of "$V/$f.json")" ] || fail "no avg_ts sample in $f.json"
done
B=$(med3 "$(tg_of "$V/bare-1.json")" "$(tg_of "$V/bare-2.json")" "$(tg_of "$V/bare-3.json")")
T=$(med3 "$(tg_of "$V/tui-1.json")" "$(tg_of "$V/tui-2.json")" "$(tg_of "$V/tui-3.json")")
awk -v b="$B" -v t="$T" 'BEGIN {
  if (b <= 0 || t <= 0) { print "CONTAMINATION-FAIL: empty sample"; exit 1 }
  r = (t > b) ? t / b : b / t
  printf "bare tg=%.2f tok/s  tui-idle tg=%.2f tok/s  ratio=%.3f\n", b, t, r
  exit (r < 1.03) ? 0 : 1
}' || fail "contamination gate (3% tolerance)"
echo "CONTAMINATION-OK"
echo "validation complete -> $V"
echo "operator live test: $BUILD/bench-tui --config-dir $V/cfg --out-dir $V/runs --bin-dir $LLAMA --models-dir $HOME/models"
