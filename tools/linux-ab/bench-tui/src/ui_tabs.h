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
  std::string pending_cmd, pending_dir;
  int last_agent_exit = -1;
  std::string notice;
};

std::shared_ptr<cpptui::Widget> build_runs_tab(cpptui::App &app, UiState &u);
std::shared_ptr<cpptui::Widget> build_system_tab(cpptui::App &app, UiState &u);
std::shared_ptr<cpptui::Widget> build_agents_tab(cpptui::App &app, UiState &u);
