#!/bin/bash
# bench-tui.tmux.sh — layout: window 0 bench-tui, 1 kimi, 2 codex.
# Usage: bench-tui.tmux.sh [session-name] [repo-dir]
set -euo pipefail
SESSION=${1:-bench}
REPO=${2:-$HOME/beyondvram}
HERE=$(cd "$(dirname "$0")" && pwd)
tmux new-session -d -s "$SESSION" -c "$REPO" -n tui "$HERE/build/bench-tui"
tmux new-window -t "$SESSION" -n kimi -c "$REPO" "kimi; exec bash"
tmux new-window -t "$SESSION" -n codex -c "$REPO" "codex; exec bash"
tmux attach -t "$SESSION"
