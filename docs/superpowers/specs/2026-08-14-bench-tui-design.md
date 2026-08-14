# bench-tui design — bench-OS control center TUI

Date: 2026-08-14. Status: approved design, pre-implementation.

## Purpose

The Debian bench box (experiment #6 secondary testbench, see `tools/linux-ab/SETUP.md`) currently has no
operator interface: bench runs are long (multi-hour llama-bench A/B rounds), progress is raw scrolling
stdout, and there is no at-a-glance view of GPU/RAM/disk state while a run executes. bench-tui is a single
C++17 terminal application providing three tabs: a bench **Runs** control center, a **System** telemetry
dashboard, and an **Agents** launcher for interactive CLI agents (Kimi CLI, Codex CLI, plus a plain shell).

The TUI is an operator convenience, never a required component: every action it can trigger remains
reproducible by hand with the same binaries and flags, and all run artifacts land in the repo's existing
results layout.

## Stack decision

- C++17, single binary, built with CMake.
- UI library: [cpp-tui](https://github.com/jonoton/cpp-tui) — single-header, MIT-licensed
  ([LICENSE](https://raw.githubusercontent.com/jonoton/cpp-tui/main/LICENSE)), actively maintained as of
  2026-08. The header is **vendored** at `tools/linux-ab/bench-tui/third_party/cpptui.hpp` with its
  copyright notice intact and the pinned upstream commit recorded in the local README, so builds are
  offline and reproducible. No new apt packages beyond what SETUP.md already installs
  (`build-essential`, `cmake`, `git`, NVIDIA driver).
- Alternatives considered and rejected: Python + Textual (off the minimal-bench-box discipline, heavier
  runtime, pip deps); bash + watch/tmux (not a real TUI, no charts, brittle).

## Approved scope decisions

From the brainstorming Q&A (2026-08-14):

1. Purpose: **both** bench companion and system dashboard, tabbed in one app.
2. Control model: **full control center** — presets, queue, edit flags in-app, start/stop runs, model
   file selection — plus the Agents tab.
3. Agent integration: **suspend/shell-out** (approach A). Selecting an agent suspends the TUI, hands the
   fullscreen terminal to the agent process, and resumes on exit. True embedded PTY tabs (libvterm +
   forkpty) are explicitly deferred to a possible v2. A tmux layout script is shipped as a bonus.

## Non-goals (YAGNI, v1)

- No embedded terminal emulator widget (no libvterm, no VT parsing).
- No model downloading or registry: "model management" means scanning a configured directory for local
  GGUF files (name, size) and picking one in the preset editor. Downloads stay manual.
- No per-token live throughput fabrication: llama-bench emits results only at the end of each test, so
  mid-test liveness comes from the GPU telemetry strip; the tok/s sparkline advances per completed test.
- No browsing of the repo's historical results tree; the results table covers runs the TUI launched
  (sourced from `results/bench-tui/runs.jsonl`), nothing else.
- No mutable GPU control (clocks, power limits, fan): all telemetry is strictly read-only.
- No Windows support target. cpp-tui is cross-platform, but bench-tui's collectors read `/proc` and NVML;
  the supported targets are the WSL2 testbench (dev) and the Debian bench box (production).

## Architecture

Single process, three layers:

```
tools/linux-ab/bench-tui/
  CMakeLists.txt
  third_party/cpptui.hpp        # vendored, MIT, pinned commit noted
  src/main.cpp                  # app wiring, tabs, keybindings, theme
  src/collectors.{h,cpp}        # NVML/nvidia-smi, /proc/stat, /proc/meminfo, /proc/diskstats
  src/run_manager.{h,cpp}       # preset queue, child spawn, log capture, record keeping
  src/config.{h,cpp}            # presets.json / agents.json load + save (minimal JSON, no new deps)
  src/ui_tabs.{h,cpp}           # Runs / System / Agents tab construction
  bench-tui.tmux.sh             # bonus layout: window 0 bench-tui, 1 kimi, 2 codex
```

- UI thread: cpp-tui `App` with a `Tabs` root (Runs, System, Agents), dark theme, global keys
  (`1/2/3` tab switch, `q` quit with confirm while a run is active).
- Collector worker threads poll gently (1–2 s) into small mutex-guarded state structs; the UI redraws via
  cpp-tui's thread-safe `update()`/`post()`.
- One run-manager worker thread owns the queue and all child processes. It is the only component that
  writes outside the app (logs + records under `results/bench-tui/`).
- Config files (`presets.json`, `agents.json`) live beside the binary by default; `--config-dir <path>`
  overrides. The models directory scanned by the preset editor defaults to `~/models` and is itself a
  config entry.

## Components

### Runs tab

- **Preset list**: entries from `presets.json`; a preset is `{ name, model_path, binary, args, repeats }`.
  `binary` is one of the known b10355 binaries (llama-bench, llama-completion, llama-server); `args` is
  the full flag string exactly as passed on the command line.
- **Preset editor**: form built from cpp-tui Input/NumberInput/Dropdown widgets; the model field offers a
  dropdown populated by scanning a configured models directory for `*.gguf`. Saves back to `presets.json`.
- **Queue**: add selected preset (with repeat count), remove while idle, start/stop. One run at a
  time — concurrent runs are refused (single GPU). Reordering is out of scope for v1.
- **Live view**: current preset and arm index, elapsed time, ETA estimated from completed repetitions of
  the same preset, tok/s sparkline fed by each completed test result, scrollable log tail of the active
  child.
- **Results table**: parsed JSON results of TUI-launched runs (test time, model, flags, pp/tg tok/s),
  newest first.

### System tab

- GPU panel: utilization %, VRAM used/total, temperature, power draw. Primary source: NVML via
  `dlopen("libnvidia-ml.so.1")`; fallback: spawning `nvidia-smi --query-gpu=... --format=csv,noheader`;
  if neither exists the panel shows "unavailable" and nothing is fabricated.
- CPU panel: per-core busy % from `/proc/stat` deltas, plus loadavg.
- Memory panel: RAM and swap total/available from `/proc/meminfo` (relevant to the planned 64 GiB upgrade
  and the gpt-oss-120b phase).
- Disk panel: per-device read/write KB/s from `/proc/diskstats` deltas, plus free space on `/` (statvfs).
- Sparkline history (last ~120 samples) for GPU util, VRAM, RAM, disk throughput.

### Agents tab

- Entries from `agents.json`: `{ name, command, workdir }`. Shipped defaults: `kimi`, `codex`, and
  `$SHELL`, all with workdir = the repo clone.
- Activation flow: cpp-tui terminal teardown → `fork`/`execvp` the command (inheriting the controlling
  terminal) → `waitpid` → TUI resume with full redraw. The child's exit code is shown in a status line.
  This is the same suspend pattern used by htop/ranger/mc and imposes no emulator code.
- `bench-tui.tmux.sh`: optional session script arranging window 0 = bench-tui, 1 = kimi, 2 = codex.

### Run manager

- Pop preset → assemble the exact command line → spawn child in its own process group (`fork` + `setsid`
  + `execvp`), stdout+stderr tee'd to `results/bench-tui/<UTC-timestamp>-<label>.log`.
- On child exit: parse any JSON output the binary wrote, append one record to
  `results/bench-tui/runs.jsonl` with `{ utc_start, utc_end, label, model_path, binary, args, exit_code,
  log_path, output_json_path, tg_ts, pp_ts }` (null fields when not applicable).
- Stop = SIGTERM to the child's process group; a second SIGKILL escalation follows after a grace period
  if the child ignores SIGTERM.
- Pre-flight guard: before starting a run, scan for running llama/llama-bench/llama-server processes
  (Linux analog of `tools/build-scripts/kill-stale-llama.ps1`); refuse to start while VRAM is occupied,
  showing the offending PIDs.

## Measurement-safety rules (hard)

1. While a run is active, all collectors drop to quiet mode (≥5 s intervals) and a
   `MEASUREMENT IN PROGRESS` banner is shown.
2. No second run can start while one is active; quitting the TUI during a run requires explicit
   confirmation and never kills the child implicitly — detach or terminate is an explicit choice.
   Ctrl+C quits immediately without the confirm bar; an active run is detached, never killed.
3. Telemetry is read-only; the app never calls any mutable NVML/nvidia-smi path.
4. Runs are launched with the same pinned b10355 binaries and the same flag conventions as the existing
   scripts, so TUI-launched results remain comparable with every prior recorded run.
5. Ship gate: a paired contamination check — the same short bench config run with the TUI active vs
   absent must agree within run-to-run noise; otherwise the TUI does not ship to the bench box.

## Error handling

- Missing NVML and nvidia-smi → GPU panel "unavailable"; never invented values (record-don't-invent).
- Missing or invalid `model_path` → preset refused at queue time with the reason displayed.
- Child spawn failure → queue entry recorded as failed in `runs.jsonl`, log retained, and the queue
  continues with the next entry.
- Non-zero child exit → recorded as-is in `runs.jsonl`; log tail highlights the failure.
- Malformed or absent JSON output → raw log remains viewable; record fields stay null.
- Corrupt `presets.json`/`agents.json` → app starts with built-in defaults and reports the parse error;
  the bad file is preserved, never overwritten silently.

## Data flow

```
/proc, NVML/nvidia-smi ──▶ collector threads ─▶ state structs (mutex) ─▶ UI redraw (update/post)
child stdout/stderr ──▶ tee to .log ─▶ line ring buffer ─▶ live tail widget
child exit ──▶ parse JSON outputs ──▶ runs.jsonl append + results table row
presets.json / agents.json ◀──▶ config load at start, save on edit
```

## Testing and validation (WSL2 first)

1. Build in the existing WSL2 testbench with its stock toolchain (`g++`, `cmake`).
2. Smoke: tabs render and switch; telemetry cross-checked against `nvidia-smi` and `htop`; Agents tab
   suspend/resume into bash and back; quit-confirm behavior during an active run.
3. Tiny real run: queue a short llama-bench config against a small local GGUF; verify log capture, live
   view updates, stop-via-SIGTERM, and the results table row.
4. Contamination check: paired A/B of the same short config, TUI active vs absent; tok/s must agree
   within noise before the app is allowed near a measured run. The battery gates the idle-TUI case;
   the managed-run case is covered by the operator live test in item 5 (which queues and starts the
   smoke preset), and any oddity there blocks the `tools/linux-ab/SETUP.md` install section.
5. Rendered screenshots are posted for review, then a hands-on live test in WSL2 by the operator.
   Only after that explicit validation does `tools/linux-ab/SETUP.md` gain a bench-tui install section
   for the Debian bench box.

## Estimated size

Five source files plus one vendored header; under roughly 1200 lines of project C++. No new system
packages, no network at build or run time.

## References

- cpp-tui: https://github.com/jonoton/cpp-tui (MIT, single header, C++17)
- Bench-box provisioning: `tools/linux-ab/SETUP.md`
- Measurement discipline: `AGENTS.md` conventions section; `docs/system-characterization.md`
