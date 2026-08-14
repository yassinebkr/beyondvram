# bench-tui

Bench-OS control-center TUI: bench Runs control, System telemetry, Agents launcher.
Design: docs/superpowers/specs/2026-08-14-bench-tui-design.md.

A Windows checkout can give the shell scripts CRLF line endings; if so, run
`sed -i 's/\r$//' validate_wsl.sh bench-tui.tmux.sh quickstart.sh` once first
(assumes cwd is the bench-tui dir).

## Quick start (any Linux or WSL)

    tools/linux-ab/bench-tui/quickstart.sh

Builds into `tools/linux-ab/bench-tui/build/` and runs the TUI. Requires only
g++ (C++17), cmake, and make.

## Run

    build/bench-tui [--config-dir DIR] [--out-dir DIR] [--bin-dir DIR] [--models-dir DIR]

Defaults: config `./bench-tui-config`, run output `./results/bench-tui`, llama binaries
`/opt/llama-b10355`, models `~/models`.

## Keys

Global: `1`/`2`/`3` switch tabs, `q` quits (confirm bar while a run is active: `s` stop and
quit, `d` detach and quit, `c` cancel). Ctrl+C quits immediately; an active run is detached,
never killed. A one-line help bar above the status line lists the keys of the current tab.

Runs tab: `a` add preset, `e` edit selected, `x` delete selected (no confirm), `n` queue
selected, `u` dequeue selected, `s` start the queue, `t` stop the active run. `Enter` on the
preset table also queues the selected preset. Queue appends the selected preset to the run
queue; dequeue removes the selected queued entry. The buttons row carries the same letters in
brackets. Runs keys fire only on the Runs tab and stand down while the quit-confirm bar is
open. While the preset editor is open, letter keys type into the form instead of triggering
shortcuts: Tab moves focus between the fields and the save/cancel buttons, Enter presses the
focused button.

Agents tab: `Enter` launches the selected agent fullscreen; exiting the agent returns to the
TUI. Agents are added by hand in `agents.json`, not in the TUI.

Mouse reporting comes from the terminal, not the TUI: it works in WSL and desktop terminals,
while a raw Linux console has no mouse unless gpm is installed. Every action also has a key.

## Agents and API keys

Agents are shell commands read from `agents.json` (`name`, `command`, `workdir`); the TUI
never stores credentials. Each agent CLI authenticates itself through its own config or
login flow - for kimi, run its `/login` flow once inside the shell agent, or export the
CLI's documented API-key variable in `~/.bashrc`. The agent process inherits the TUI's
environment, so an export in `~/.bashrc` is enough.

## Tests

    build/bench-tui-tests

## Validation

`validate_wsl.sh` runs build + unit tests + pty smoke + the measurement-contamination A/B
(same tiny CPU-only llama-bench config, TUI idle vs absent, 3% tolerance gate). Override
`BUILD`, `LLAMA`, `V` env vars to change paths. The battery gates the idle-TUI case only;
the managed-run case (the TUI driving the bench, 100 ms log tail + 500 ms UI tick) is covered
by the operator live test, which queues and starts the smoke preset, and any oddity there
blocks the SETUP.md install section.

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
