# bench-tui

Bench-OS control-center TUI (Runs / System / Agents). Design:
docs/superpowers/specs/2026-08-14-bench-tui-design.md.

## Quick start (any Linux or WSL)

    tools/linux-ab/bench-tui/quickstart.sh

Builds into `tools/linux-ab/bench-tui/build/` and runs the TUI. Requires only
g++ (C++17), cmake, and make. `q` quits.

## Vendored dependency

`third_party/cpptui.hpp` — cpp-tui @ 9543ee3c056583eea1fc44491c6c240f0df0b570 (MIT),
sha256 9942394fd8b52f9948313c160a45c360d503c34e52df0626787e36da6927137e, 570572 bytes.
Re-vendor:

    curl -sL -o third_party/cpptui.hpp \
      https://raw.githubusercontent.com/jonoton/cpp-tui/9543ee3c056583eea1fc44491c6c240f0df0b570/cpptui.hpp
