# bench-tui

Bench-OS control-center TUI: bench Runs control, System telemetry, Agents launcher.
Design: docs/superpowers/specs/2026-08-14-bench-tui-design.md.

## Quick start (any Linux or WSL)

    tools/linux-ab/bench-tui/quickstart.sh

Builds into `tools/linux-ab/bench-tui/build/` and runs the TUI. Requires only
g++ (C++17), cmake, and make.

## Run

    build/bench-tui [--config-dir DIR] [--out-dir DIR] [--bin-dir DIR] [--models-dir DIR]

Defaults: config `./bench-tui-config`, run output `./results/bench-tui`, llama binaries
`/opt/llama-b10355`, models `~/models`.

Keys: `1`/`2`/`3` switch tabs, `q` quits (confirm bar while a run is active: `s` stop and
quit, `d` detach and quit, `c` cancel). While the preset editor is open, letter keys type
into the form instead of triggering shortcuts. `Enter` on the Agents tab launches the
selected agent fullscreen; exiting the agent returns to the TUI. Ctrl+C quits immediately;
an active run is detached, never killed.

The Runs tab buttons row is add / edit / del / queue / dequeue / start / stop: queue
appends the selected preset to the run queue, dequeue removes the selected queued entry.

## Tests

    build/bench-tui-tests

## Validation

`validate_wsl.sh` runs build + unit tests + pty smoke + the measurement-contamination A/B
(same tiny CPU-only llama-bench config, TUI idle vs absent, 3% tolerance gate). Override
`BUILD`, `LLAMA`, `V` env vars to change paths. When the checkout has CRLF line endings,
run `sed -i 's/\r$//' validate_wsl.sh bench-tui.tmux.sh quickstart.sh` once first.

## Vendored dependency

`third_party/cpptui.hpp` — cpp-tui @ 9543ee3c056583eea1fc44491c6c240f0df0b570 (MIT),
sha256 9942394fd8b52f9948313c160a45c360d503c34e52df0626787e36da6927137e, 570572 bytes.
Re-vendor:

    curl -sL -o third_party/cpptui.hpp \
      https://raw.githubusercontent.com/jonoton/cpp-tui/9543ee3c056583eea1fc44491c6c240f0df0b570/cpptui.hpp

## Record keeping

Every completed child appends one JSON line to `<out-dir>/runs.jsonl`; stderr lands in
`<ts>-<label>-r<n>-<seq>.log`, raw stdout in `<ts>-<label>-r<n>-<seq>.out`. Stopped runs are
recorded with `note: "stopped"`, detached runs with `note: "detached"` (no exit code).

## tmux layout

`bench-tui.tmux.sh [session] [repo-dir]` opens window 0 = bench-tui, 1 = kimi, 2 = codex.
