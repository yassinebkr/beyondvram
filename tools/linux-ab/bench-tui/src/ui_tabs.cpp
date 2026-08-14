#include "ui_tabs.h"
#include "format.h"
#include "fsutil.h"
#include "cpptui.hpp"
#include <cstdio>
#include <string>
#include <vector>
using namespace cpptui;

namespace {
std::vector<StyledText> to_styled(const std::vector<std::string> &v) {
  std::vector<StyledText> out;
  out.reserve(v.size());
  for (const auto &s : v) out.emplace_back(s);
  return out;
}
std::string join_last(const std::vector<std::string> &v, size_t n) {
  size_t start = v.size() > n ? v.size() - n : 0;
  std::string r;
  for (size_t i = start; i < v.size(); ++i) {
    r += v[i];
    r += '\n';
  }
  return r;
}
}  // namespace

std::shared_ptr<Widget> build_runs_tab(App &app, UiState &u) {
  auto root = std::make_shared<Vertical>();

  auto body = std::make_shared<Horizontal>();
  root->add(body);

  // left column: presets, actions, editor
  auto left = std::make_shared<Vertical>();
  left->min_width = 34;
  body->add(left);
  left->add(std::make_shared<Label>("Presets"));
  auto presets_table = std::make_shared<TableScrollable>();
  presets_table->columns = to_styled({"name", "binary", "reps"});
  presets_table->min_height = 6;
  left->add(presets_table);

  auto refresh_presets = [&u, presets_table]() {
    presets_table->rows.clear();
    for (const auto &p : u.presets)
      presets_table->add_row({p.name, p.binary, std::to_string(p.repeats)});
  };
  refresh_presets();

  auto actions = std::make_shared<Horizontal>();
  auto add_btn = std::make_shared<Button>("add");
  auto edit_btn = std::make_shared<Button>("edit");
  auto del_btn = std::make_shared<Button>("del");
  auto queue_btn = std::make_shared<Button>("queue");
  auto dequeue_btn = std::make_shared<Button>("dequeue");
  auto start_btn = std::make_shared<Button>("start");
  auto stop_btn = std::make_shared<Button>("stop");
  for (auto b : {add_btn, edit_btn, del_btn, queue_btn, dequeue_btn, start_btn, stop_btn})
    actions->add(b);
  left->add(actions);

  // editor panel (hidden unless add/edit)
  auto editor = std::make_shared<Vertical>();
  editor->visible = false;
  auto ed_name = std::make_shared<Input>();
  ed_name->placeholder = "name";
  auto ed_model = std::make_shared<Dropdown>(&app);
  auto ed_binary = std::make_shared<Dropdown>(&app);
  ed_binary->set_options(to_styled({"llama-bench", "llama-completion", "llama-server"}));
  auto ed_args = std::make_shared<Input>();
  ed_args->placeholder = "-m <model> -ngl 48 --n-cpu-moe 33 -p 128 -n 32 -r 5 -o json";
  auto ed_reps = std::make_shared<NumberInput>(1);
  ed_reps->min_value = 1;
  ed_reps->max_value = 100000;  // default max 100 would silently rewrite larger saved repeats
  editor->add(std::make_shared<Label>("name:"));
  editor->add(ed_name);
  editor->add(std::make_shared<Label>("model (recorded as model_path; args must carry the matching -m):"));
  editor->add(ed_model);
  editor->add(std::make_shared<Label>("binary:"));
  editor->add(ed_binary);
  editor->add(std::make_shared<Label>("args (whitespace-separated, no quoting):"));
  editor->add(ed_args);
  editor->add(std::make_shared<Label>("repeats:"));
  editor->add(ed_reps);
  auto save_btn = std::make_shared<Button>("save");
  auto cancel_btn = std::make_shared<Button>("cancel");
  auto ed_buttons = std::make_shared<Horizontal>();
  ed_buttons->add(save_btn);
  ed_buttons->add(cancel_btn);
  editor->add(ed_buttons);
  left->add(editor);

  auto editing_idx = std::make_shared<int>(-1);
  auto model_opts = std::make_shared<std::vector<std::string>>();

  auto open_editor = [&u, editor, ed_name, ed_model, ed_binary, ed_args, ed_reps, editing_idx,
                      model_opts](int idx) mutable {
    *editing_idx = idx;
    *model_opts = scan_gguf(u.models_dir);
    std::vector<std::string> labels = *model_opts;
    if (labels.empty()) labels.push_back("(no .gguf under models dir)");
    ed_model->set_options(to_styled(labels));
    ed_model->selected_index = -1;
    ed_binary->selected_index = -1;
    if (idx >= 0 && idx < (int)u.presets.size()) {
      const Preset &p = u.presets[idx];
      ed_name->set_value(p.name);
      ed_args->set_value(p.args);
      ed_reps->set_value(p.repeats);
      for (size_t k = 0; k < model_opts->size(); ++k)
        if ((*model_opts)[k] == p.model_path) ed_model->selected_index = (int)k;
      static const char *bins[] = {"llama-bench", "llama-completion", "llama-server"};
      for (int k = 0; k < 3; ++k)
        if (p.binary == bins[k]) ed_binary->selected_index = k;
    } else {
      ed_name->set_value("");
      ed_args->set_value("");
      ed_reps->set_value(1);
    }
    // the editor and the quit-confirm bar never coexist
    u.confirming = false;
    if (u.quitbar_label) u.quitbar_label->visible = false;
    editor->visible = true;
    u.editing = true;
  };

  save_btn->set_on_click([&u, editor, ed_name, ed_model, ed_binary, ed_args, ed_reps, editing_idx,
                          model_opts, presets_table, refresh_presets]() mutable {
    Preset p;
    p.name = ed_name->get_value();
    int mi = ed_model->selected_index;
    p.model_path = (mi >= 0 && mi < (int)model_opts->size()) ? (*model_opts)[mi] : "";
    static const char *bins[] = {"llama-bench", "llama-completion", "llama-server"};
    int bi = ed_binary->selected_index;
    p.binary = (bi >= 0 && bi < 3) ? bins[bi] : "";
    p.args = ed_args->get_value();
    p.repeats = ed_reps->get_value();
    if (p.name.empty() || p.binary.empty() || p.args.empty() || p.repeats < 1) {
      u.notice = "preset invalid: name, binary, args and repeats>=1 are required";
      return;
    }
    if (p.model_path.empty() || !file_exists(p.model_path)) {
      u.notice = "preset invalid: model file not found";
      return;
    }
    if (*editing_idx >= 0 && *editing_idx < (int)u.presets.size()) u.presets[*editing_idx] = p;
    else u.presets.push_back(p);
    u.notice = save_presets(u.presets_path, u.presets) ? "presets saved" : "preset save failed";
    refresh_presets();
    editor->visible = false;
    u.editing = false;
    presets_table->set_focus(true);
  });
  cancel_btn->set_on_click([&u, editor, presets_table]() {
    editor->visible = false;
    u.editing = false;
    presets_table->set_focus(true);
  });

  add_btn->set_on_click([open_editor]() mutable { open_editor(-1); });
  edit_btn->set_on_click([open_editor, presets_table]() mutable {
    open_editor(presets_table->selected_index);
  });
  del_btn->set_on_click([&u, presets_table, refresh_presets]() mutable {
    int idx = presets_table->selected_index;
    if (idx < 0 || idx >= (int)u.presets.size()) return;
    u.presets.erase(u.presets.begin() + idx);
    u.notice = save_presets(u.presets_path, u.presets) ? "presets saved" : "preset save failed";
    refresh_presets();
  });
  auto queue_cb = [&u](int idx) {
    if (idx < 0 || idx >= (int)u.presets.size()) return;
    if (!file_exists(u.presets[idx].model_path)) {
      u.notice = "refused: model file missing: " + u.presets[idx].model_path;
      return;
    }
    u.runs->enqueue({u.presets[idx]});
    u.notice = "queued " + u.presets[idx].name;
  };
  queue_btn->set_on_click([queue_cb, presets_table]() { queue_cb(presets_table->selected_index); });
  presets_table->on_submit = queue_cb;
  start_btn->set_on_click([&u]() {
    std::vector<int> bad;
    if (u.runs->preflight(bad) > 0) {
      u.notice = "VRAM busy: llama-* pid(s):";
      for (int pid : bad) u.notice += " " + std::to_string(pid);
      return;
    }
    u.notice = u.runs->start() ? "run started" : "start refused: queue empty or already running";
  });
  stop_btn->set_on_click([&u]() {
    u.runs->stop();
    u.notice = "stop requested";
  });

  // right column: queue, live run, results
  auto right = std::make_shared<Vertical>();
  body->add(right);
  right->add(std::make_shared<Label>("Queue"));
  auto queue_table = std::make_shared<TableScrollable>();
  queue_table->columns = to_styled({"preset", "reps"});
  queue_table->min_height = 4;
  right->add(queue_table);
  dequeue_btn->set_on_click([&u, queue_table]() {
    if (u.runs->status().active) {
      u.notice = "queue locked while a run is active";
      return;
    }
    u.notice = u.runs->remove_queued(queue_table->selected_index) ? "dequeued"
                                                                  : "dequeue: nothing selected";
  });
  auto live = std::make_shared<Label>("idle");
  right->add(live);
  auto tps = std::make_shared<Sparkline>();
  tps->auto_scale = true;
  tps->show_label = true;
  tps->label_format = "%.1f tok/s";
  tps->min_height = 3;
  right->add(tps);
  auto tail = std::make_shared<Paragraph>();
  tail->min_height = 6;
  right->add(tail);
  right->add(std::make_shared<Label>("Results"));
  auto results = std::make_shared<TableScrollable>();
  results->columns = to_styled({"time", "label", "tg", "pp", "exit"});
  results->min_height = 6;
  right->add(results);

  auto last_qs = std::make_shared<std::string>();
  auto last_n = std::make_shared<size_t>((size_t)-1);
  app.add_timer(500, [&app, &u, live, tps, tail, results, queue_table, last_qs, last_n]() {
    RunStatus st = u.runs->status();
    if (u.banner_label) u.banner_label->visible = st.active;
    if (!st.active && u.confirming) {  // run ended on its own: drop the stale confirm bar
      u.confirming = false;
      if (u.quitbar_label) u.quitbar_label->visible = false;
    }
    if (u.status_label) u.status_label->set_text(u.notice);
    u.runner->set_interval_ms(st.active ? 5000 : 1500);
    if (st.active)
      live->set_text("run: " + st.label + "  arm " + std::to_string(st.arm_index) + "/" +
                     std::to_string(st.arm_count) + "  elapsed " + format_duration(st.elapsed_s) +
                     "  eta " + format_duration(st.eta_s));
    else
      live->set_text("idle");
    auto q = u.runs->queue_snapshot();
    std::string qs;
    for (const auto &item : q)
      qs += item.preset.name + ";" + std::to_string(item.preset.repeats) + ";";
    if (qs != *last_qs) {
      *last_qs = qs;
      queue_table->rows.clear();
      for (const auto &item : q)
        queue_table->add_row({item.preset.name, std::to_string(item.preset.repeats)});
    }
    auto recs = u.runs->records();
    if (recs.size() != *last_n) {
      *last_n = recs.size();
      tps->data.clear();
      results->rows.clear();
      for (const auto &r : recs)
        if (r.tg_ts >= 0) tps->data.push_back((float)r.tg_ts);
      for (auto it = recs.rbegin(); it != recs.rend(); ++it) {
        const RunRecord &r = *it;
        results->add_row({r.utc_start.size() >= 19 ? r.utc_start.substr(11, 8) : r.utc_start,
                          r.label, r.tg_ts >= 0 ? format_ts(r.tg_ts) : "-",
                          r.pp_ts >= 0 ? format_ts(r.pp_ts) : "-", std::to_string(r.exit_code)});
      }
    }
    tail->set_text(join_last(u.runs->tail_snapshot(), 8));
    app.update();
  });
  return root;
}

std::shared_ptr<Widget> build_system_tab(App &app, UiState &u) {
  auto root = std::make_shared<Vertical>();
  auto gpu_label = std::make_shared<Label>("GPU: ...");
  root->add(gpu_label);
  auto vram_bar = std::make_shared<ProgressBar>();
  vram_bar->min_height = 1;
  root->add(vram_bar);
  auto gpu_spark = std::make_shared<Sparkline>();
  gpu_spark->min_val = 0;
  gpu_spark->max_val = 100;
  gpu_spark->show_label = true;
  gpu_spark->label_format = "GPU %.0f%%";
  gpu_spark->min_height = 3;
  root->add(gpu_spark);
  auto mem_label = std::make_shared<Label>("RAM: ...");
  root->add(mem_label);
  auto ram_bar = std::make_shared<ProgressBar>();
  ram_bar->min_height = 1;
  root->add(ram_bar);
  auto ram_spark = std::make_shared<Sparkline>();
  ram_spark->min_val = 0;
  ram_spark->max_val = 100;
  ram_spark->show_label = true;
  ram_spark->label_format = "RAM %.0f%%";
  ram_spark->min_height = 3;
  root->add(ram_spark);
  root->add(std::make_shared<Label>("CPU per core:"));
  auto cpu_table = std::make_shared<TableScrollable>();
  cpu_table->columns = to_styled({"core", "busy"});
  cpu_table->min_height = 8;
  root->add(cpu_table);
  root->add(std::make_shared<Label>("Disk:"));
  auto disk_table = std::make_shared<TableScrollable>();
  disk_table->columns = to_styled({"device", "read", "write"});
  disk_table->min_height = 5;
  root->add(disk_table);
  auto disk_spark = std::make_shared<Sparkline>();
  disk_spark->auto_scale = true;
  disk_spark->show_label = true;
  disk_spark->label_format = "disk %.1f MB/s";
  disk_spark->min_height = 3;
  root->add(disk_spark);

  app.add_timer(1000, [&app, &u, gpu_label, vram_bar, gpu_spark, mem_label, ram_bar, ram_spark,
                       cpu_table, disk_table, disk_spark]() {
    CollectorSnapshot s = snapshot_from(u.cols);
    if (s.gpu.available) {
      gpu_label->set_text("GPU " + std::to_string(s.gpu.util_pct) + "%  " +
                          std::to_string(s.gpu.temp_c) + "C  " + std::to_string(s.gpu.power_w) +
                          " W  VRAM " + format_gib((double)s.gpu.vram_used_mb) + "/" +
                          format_gib((double)s.gpu.vram_total_mb) + "  (" + s.gpu.source + ")");
      vram_bar->value =
          s.gpu.vram_total_mb ? (float)s.gpu.vram_used_mb / (float)s.gpu.vram_total_mb : 0.0f;
    } else {
      gpu_label->set_text("GPU unavailable");
      vram_bar->value = 0.0f;
    }
    gpu_spark->data.assign(s.gpu_util_hist.begin(), s.gpu_util_hist.end());
    char load[16];
    std::snprintf(load, sizeof load, "%.2f", (double)s.cpu.load1);
    mem_label->set_text("RAM " + format_gib((double)(s.mem.total_mb - s.mem.avail_mb)) + "/" +
                        format_gib((double)s.mem.total_mb) + "  swap " +
                        format_gib((double)(s.mem.swap_total_mb - s.mem.swap_free_mb)) + "/" +
                        format_gib((double)s.mem.swap_total_mb) + "  load " + load);
    ram_bar->value =
        s.mem.total_mb ? (float)(s.mem.total_mb - s.mem.avail_mb) / (float)s.mem.total_mb : 0.0f;
    ram_spark->data.assign(s.ram_hist.begin(), s.ram_hist.end());
    cpu_table->rows.clear();
    for (size_t k = 0; k < s.cpu.per_core.size(); ++k)
      cpu_table->add_row({"cpu" + std::to_string(k), format_pct(s.cpu.per_core[k])});
    disk_table->rows.clear();
    for (size_t k = 0; k < s.disks.size() && k < 8; ++k)
      disk_table->add_row(
          {s.disks[k].device, format_mbs(s.disks[k].read_kbs), format_mbs(s.disks[k].write_kbs)});
    disk_spark->data.assign(s.disk_hist.begin(), s.disk_hist.end());
    app.update();
  });
  return root;
}

std::shared_ptr<Widget> build_agents_tab(App &app, UiState &u) {
  (void)app;
  auto root = std::make_shared<Vertical>();
  root->add(std::make_shared<Label>(
      "Agents - Enter launches fullscreen; exiting the agent returns here"));
  auto table = std::make_shared<TableScrollable>();
  table->columns = to_styled({"name", "command", "workdir"});
  for (const auto &a : u.agents) table->add_row({a.name, a.command, a.workdir});
  table->min_height = 6;
  root->add(table);
  root->add(std::make_shared<Label>(
      u.last_agent_exit >= 0 ? "last agent exit: " + std::to_string(u.last_agent_exit) : ""));
  table->on_submit = [&u](int idx) {
    if (idx < 0 || idx >= (int)u.agents.size()) return;
    u.pending_cmd = u.agents[idx].command;
    u.pending_dir = u.agents[idx].workdir;
    App::quit();
  };
  return root;
}
