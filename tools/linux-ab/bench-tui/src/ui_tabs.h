#pragma once
// The three tabs. All logic lives in the modules; this layer only wires
// widgets to state. UI refreshes off App timers; collectors and the run
// manager are never called from the render path directly.
#include "collectors.h"
#include "config.h"
#include "run_manager.h"
#include <memory>
#include <string>
#include <vector>

namespace cpptui {
class App;
class Label;
class Widget;
}  // namespace cpptui

struct UiState {
  CollectorState cols;
  RunManager *runs = nullptr;
  CollectorRunner *runner = nullptr;  // quiet-mode control
  std::vector<Preset> presets;
  std::string presets_path;
  std::vector<AgentSpec> agents;
  std::string models_dir;
  bool editing = false;               // preset editor open: global letter keys stand down
  bool confirming = false;            // quit-confirm bar open (run active)
  // Modal/status rows live in main (outside the tabs) so every tab sees them;
  // the widgets are rebuilt with each App, so the pointers are re-assigned per loop.
  std::shared_ptr<cpptui::Label> banner_label;   // MEASUREMENT IN PROGRESS
  std::shared_ptr<cpptui::Label> quitbar_label;  // quit-confirm row
  std::shared_ptr<cpptui::Label> status_label;   // notice line, rendered every timer tick
  std::string pending_cmd, pending_dir;
  std::string last_agent_exit_text;  // "last agent exit: <rc>"; empty until the first agent exits
  std::string notice;
};

std::shared_ptr<cpptui::Widget> build_runs_tab(cpptui::App &app, UiState &u);
std::shared_ptr<cpptui::Widget> build_system_tab(cpptui::App &app, UiState &u);
std::shared_ptr<cpptui::Widget> build_agents_tab(cpptui::App &app, UiState &u);
