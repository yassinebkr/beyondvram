#include "collectors.h"
#include "config.h"
#include "fsutil.h"
#include "run_manager.h"
#include "ui_tabs.h"
#include "cpptui.hpp"
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
using namespace cpptui;

// Tabs without the vendored letter shortcuts: cpptui Tabs maps n/p/[/]/ to
// next/prev tab whenever a tab button is focused, which collides with the
// Runs-tab action keys (n = queue). Digit keys 1/2/3 and the mouse stay the
// only tab switches, so the on-screen key help stays truthful.
class BenchTabs : public Tabs {
 public:
  bool on_event(const Event &event) override {
    if (event.is_key_event() && !event.ctrl && !event.alt &&
        (event.key == 'n' || event.key == 'p' || event.key == '[' || event.key == ']' ||
         event.key == '/'))
      return false;
    return Tabs::on_event(event);
  }
};

// Runs a command on the real terminal while the TUI is down (Agents tab).
// SIGINT is ignored in the parent for the duration: a Ctrl+C meant for the
// agent must not kill the TUI. Ignored dispositions survive exec, so the
// child resets SIGINT to SIG_DFL before launching the shell.
static int run_on_terminal(const std::string &cmd, const std::string &dir) {
  struct sigaction sa {}, old_sa {};
  sa.sa_handler = SIG_IGN;
  sigaction(SIGINT, &sa, &old_sa);
  pid_t pid = fork();
  if (pid == 0) {
    signal(SIGINT, SIG_DFL);
    if (!dir.empty() && chdir(dir.c_str()) != 0) _exit(127);  // bad workdir = launch failure
    execl("/bin/sh", "sh", "-c", cmd.c_str(), (char *)nullptr);
    _exit(127);
  }
  if (pid < 0) {
    sigaction(SIGINT, &old_sa, nullptr);
    return -1;
  }
  int st = 0;
  pid_t w;
  while ((w = waitpid(pid, &st, 0)) < 0 && errno == EINTR) {
  }
  sigaction(SIGINT, &old_sa, nullptr);
  if (w < 0) return -1;  // waitpid failed: st holds no valid status
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
  if (!ensure_presets_config(u.presets_path, u.presets))
    u.notice = "presets.json malformed - defaults in use, file preserved";
  std::string agents_path = config_dir + "/agents.json";
  if (!ensure_agents_config(agents_path, home + "/beyondvram", u.agents))
    u.notice = "agents.json malformed - defaults in use, file preserved";

  RunManager runs(out_dir, bin_dir);
  u.runs = &runs;
  CollectorRunner runner(u.cols);
  runner.start();
  u.runner = &runner;

  for (;;) {
    u.editing = false;     // a rebuild must not inherit a stuck modal state
    u.confirming = false;
    u.active_tab = 0;      // a fresh Tabs always starts on Runs
    App app;
    Theme::set_theme(Theme::Dark());

    // banner/quitbar/help/status live outside the tabs so they show on every tab;
    // the tab timers reach them through UiState
    auto banner = std::make_shared<Label>("MEASUREMENT IN PROGRESS - quiet polling (5 s)");
    banner->visible = false;
    u.banner_label = banner;
    auto quitbar = std::make_shared<Label>(
        "run active - [s] stop run and quit   [d] detach and quit   [c] cancel");
    quitbar->visible = false;
    u.quitbar_label = quitbar;
    auto help = std::make_shared<Label>("1/2/3 tabs | q quit");
    u.help_label = help;
    auto status = std::make_shared<Label>(u.notice);
    u.status_label = status;

    auto tabs = std::make_shared<BenchTabs>();
    tabs->add_tab("Runs", build_runs_tab(app, u));
    tabs->add_tab("System", build_system_tab(app, u));
    tabs->add_tab("Agents", build_agents_tab(app, u));
    tabs->on_change = [&u](int i) { u.active_tab = i; };

    auto root = std::make_shared<Vertical>();
    root->add(banner);
    root->add(tabs);
    root->add(quitbar);
    root->add(help);
    root->add(status);

    const bool eat = false;  // consume=false: letters must reach Input widgets when editing
    app.register_key('1', [&u, tabs] { if (!u.editing) tabs->select_tab(0); }, false, false, false, eat);
    app.register_key('2', [&u, tabs] { if (!u.editing) tabs->select_tab(1); }, false, false, false, eat);
    app.register_key('3', [&u, tabs] { if (!u.editing) tabs->select_tab(2); }, false, false, false, eat);
    app.register_key('q',
                     [&u] {
                       if (u.editing) return;
                       if (u.runs->status().active) {
                         u.confirming = true;
                         if (u.quitbar_label) u.quitbar_label->visible = true;
                       } else {
                         App::quit();
                       }
                     },
                     false, false, false, eat);
    // 's' is shared: cpptui key bindings are last-wins per key, so one handler
    // branches on state - quit-confirm stop+quit while confirming, Runs-tab
    // start otherwise (u.start_queue is set by build_runs_tab).
    app.register_key('s',
                     [&u] {
                       if (u.editing) return;
                       if (u.confirming) {
                         u.runs->stop();
                         u.confirming = false;
                         App::quit();
                         return;
                       }
                       if (u.active_tab == 0 && u.start_queue) u.start_queue();
                     },
                     false, false, false, eat);
    app.register_key('d',
                     [&u] {
                       if (u.editing) return;
                       if (!u.confirming) return;
                       u.runs->detach();
                       u.confirming = false;
                       App::quit();
                     },
                     false, false, false, eat);
    app.register_key('c',
                     [&u] {
                       if (u.editing || !u.confirming) return;
                       u.confirming = false;
                       if (u.quitbar_label) u.quitbar_label->visible = false;
                     },
                     false, false, false, eat);

    app.run(root);
    if (u.pending_cmd.empty()) break;
    std::printf("\n[bench-tui] launching: %s\n", u.pending_cmd.c_str());
    int rc = run_on_terminal(u.pending_cmd, u.pending_dir);
    u.last_agent_exit_text = "last agent exit: " + std::to_string(rc);  // rc < 0: died by signal -rc
    u.pending_cmd.clear();
    u.pending_dir.clear();
    std::printf("[bench-tui] agent exited rc=%d - press Enter to return\n", rc);
    std::string line;
    std::getline(std::cin, line);
  }
  runner.stop();
  return 0;
}
