#include "collectors.h"
#include "config.h"
#include "fsutil.h"
#include "run_manager.h"
#include "ui_tabs.h"
#include "cpptui.hpp"
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
using namespace cpptui;

// Runs a command on the real terminal while the TUI is down (Agents tab).
static int run_on_terminal(const std::string &cmd, const std::string &dir) {
  pid_t pid = fork();
  if (pid == 0) {
    if (!dir.empty() && chdir(dir.c_str()) != 0) _exit(127);  // bad workdir = launch failure
    execl("/bin/sh", "sh", "-c", cmd.c_str(), (char *)nullptr);
    _exit(127);
  }
  if (pid < 0) return -1;
  int st = 0;
  while (waitpid(pid, &st, 0) < 0 && errno == EINTR) {
  }
  if (WIFEXITED(st)) return WEXITSTATUS(st);
  if (WIFSIGNALED(st)) return -WTERMSIG(st);
  return -1;
}

int main(int argc, char **argv) {
  std::string home = getenv("HOME") ? getenv("HOME") : ".";
  std::string config_dir = "bench-tui-config";
  std::string out_dir = "results/bench-tui";
  std::string bin_dir = "/opt/llama-b10355";
  std::string models_dir = home + "/models";
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--config-dir" && i + 1 < argc) config_dir = argv[++i];
    else if (a == "--out-dir" && i + 1 < argc) out_dir = argv[++i];
    else if (a == "--bin-dir" && i + 1 < argc) bin_dir = argv[++i];
    else if (a == "--models-dir" && i + 1 < argc) models_dir = argv[++i];
    else {
      std::fprintf(stderr, "bench-tui: unknown argument: %s\n", a.c_str());
      return 2;
    }
  }
  if (!ensure_dir(config_dir) || !ensure_dir(out_dir)) {
    std::fprintf(stderr, "bench-tui: cannot create config/out directories\n");
    return 1;
  }

  UiState u;
  u.models_dir = models_dir;
  u.presets_path = config_dir + "/presets.json";
  if (!load_presets(u.presets_path, u.presets))
    u.notice = "presets.json malformed - defaults in use, file preserved";
  std::string agents_path = config_dir + "/agents.json";
  if (!load_agents(agents_path, u.agents))
    u.notice = "agents.json malformed - defaults in use, file preserved";
  if (u.agents.empty()) {
    u.agents = default_agents(home + "/beyondvram");
    save_agents(agents_path, u.agents);
  }

  RunManager runs(out_dir, bin_dir);
  u.runs = &runs;
  CollectorRunner runner(u.cols);
  runner.start();
  u.runner = &runner;

  for (;;) {
    App app;
    Theme::set_theme(Theme::Dark());
    auto tabs = std::make_shared<Tabs>();
    tabs->add_tab("Runs", build_runs_tab(app, u));
    tabs->add_tab("System", build_system_tab(app, u));
    tabs->add_tab("Agents", build_agents_tab(app, u));

    auto quitbar = std::make_shared<Label>(
        "run active - [s] stop run and quit   [d] detach and quit   [c] cancel");
    quitbar->visible = false;
    auto status = std::make_shared<Label>(u.notice);
    u.notice.clear();
    auto root = std::make_shared<Vertical>();
    root->add(tabs);
    root->add(quitbar);
    root->add(status);

    auto confirming = std::make_shared<bool>(false);
    const bool eat = false;  // consume=false: letters must reach Input widgets when editing
    app.register_key('1', [&u, tabs] { if (!u.editing) tabs->select_tab(0); }, false, false, false, eat);
    app.register_key('2', [&u, tabs] { if (!u.editing) tabs->select_tab(1); }, false, false, false, eat);
    app.register_key('3', [&u, tabs] { if (!u.editing) tabs->select_tab(2); }, false, false, false, eat);
    app.register_key('q',
                     [&u, quitbar, confirming] {
                       if (u.editing) return;
                       if (u.runs->status().active) {
                         *confirming = true;
                         quitbar->visible = true;
                       } else {
                         App::quit();
                       }
                     },
                     false, false, false, eat);
    app.register_key('s',
                     [&u, confirming] {
                       if (!*confirming) return;
                       u.runs->stop();
                       *confirming = false;
                       App::quit();
                     },
                     false, false, false, eat);
    app.register_key('d',
                     [&u, confirming] {
                       if (!*confirming) return;
                       u.runs->detach();
                       *confirming = false;
                       App::quit();
                     },
                     false, false, false, eat);
    app.register_key('c',
                     [&u, quitbar, confirming] {
                       if (u.editing || !*confirming) return;
                       *confirming = false;
                       quitbar->visible = false;
                     },
                     false, false, false, eat);

    app.run(root);
    if (u.pending_cmd.empty()) break;
    std::printf("\n[bench-tui] launching: %s\n", u.pending_cmd.c_str());
    int rc = run_on_terminal(u.pending_cmd, u.pending_dir);
    u.last_agent_exit = rc;
    u.pending_cmd.clear();
    u.pending_dir.clear();
    std::printf("[bench-tui] agent exited rc=%d - press Enter to return\n", rc);
    std::string line;
    std::getline(std::cin, line);
  }
  runner.stop();
  return 0;
}
