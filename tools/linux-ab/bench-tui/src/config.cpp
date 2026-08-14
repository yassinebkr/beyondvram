#include "config.h"
#include "fsutil.h"
#include <algorithm>

static std::string jstr(const JsonValue &o, const char *k) {
  const JsonValue *v = o.find(k);
  return (v && v->type == JsonValue::Type::Str && v->s.find('\0') == std::string::npos) ? v->s : "";
}
static long long jint(const JsonValue &o, const char *k, long long dflt) {
  const JsonValue *v = o.find(k);
  return (v && v->type == JsonValue::Type::Int) ? v->i : dflt;
}

bool load_presets(const std::string &path, std::vector<Preset> &out) {
  out.clear();
  if (!file_exists(path)) return true;
  bool ok = false;
  std::string text = slurp(path, ok);
  JsonValue v;
  if (!ok || !json_parse(text, v) || v.type != JsonValue::Type::Arr) {
    out.clear();
    return false;
  }
  for (const auto &e : v.arr) {
    Preset p;
    p.name = jstr(e, "name");
    p.model_path = jstr(e, "model_path");
    p.binary = jstr(e, "binary");
    p.args = jstr(e, "args");
    p.repeats = (int)std::min<long long>(std::max<long long>(jint(e, "repeats", 1), 1), 1000000);
    if (p.name.empty() || p.binary.empty()) continue;  // entries without identity are dropped
    out.push_back(std::move(p));
  }
  return true;
}

bool save_presets(const std::string &path, const std::vector<Preset> &v) {
  std::vector<JsonValue> arr;
  for (const auto &p : v) {
    std::map<std::string, JsonValue> m;
    m["name"] = JsonValue::make_str(p.name);
    m["model_path"] = JsonValue::make_str(p.model_path);
    m["binary"] = JsonValue::make_str(p.binary);
    m["args"] = JsonValue::make_str(p.args);
    m["repeats"] = JsonValue::make_int(p.repeats);
    arr.push_back(JsonValue::make_obj(std::move(m)));
  }
  return write_file(path, json_stringify(JsonValue::make_arr(std::move(arr))) + "\n");
}

bool load_agents(const std::string &path, std::vector<AgentSpec> &out) {
  out.clear();
  if (!file_exists(path)) return true;
  bool ok = false;
  std::string text = slurp(path, ok);
  JsonValue v;
  if (!ok || !json_parse(text, v) || v.type != JsonValue::Type::Arr) {
    out.clear();
    return false;
  }
  for (const auto &e : v.arr) {
    AgentSpec a;
    a.name = jstr(e, "name");
    a.command = jstr(e, "command");
    a.workdir = jstr(e, "workdir");
    if (a.name.empty() || a.command.empty()) continue;
    out.push_back(std::move(a));
  }
  return true;
}

bool save_agents(const std::string &path, const std::vector<AgentSpec> &v) {
  std::vector<JsonValue> arr;
  for (const auto &a : v) {
    std::map<std::string, JsonValue> m;
    m["name"] = JsonValue::make_str(a.name);
    m["command"] = JsonValue::make_str(a.command);
    m["workdir"] = JsonValue::make_str(a.workdir);
    arr.push_back(JsonValue::make_obj(std::move(m)));
  }
  return write_file(path, json_stringify(JsonValue::make_arr(std::move(arr))) + "\n");
}

std::vector<AgentSpec> default_agents(const std::string &repo_dir) {
  return {{"kimi", "kimi", repo_dir},
          {"codex", "codex", repo_dir},
          {"shell", "${SHELL:-/bin/bash}", repo_dir}};
}

bool ensure_presets_config(const std::string &path, std::vector<Preset> &out) {
  out.clear();
  if (file_exists(path)) return load_presets(path, out);  // false: malformed, file preserved
  save_presets(path, out);                                // absent: write empty defaults once
  return true;
}

bool ensure_agents_config(const std::string &path, const std::string &repo_dir,
                          std::vector<AgentSpec> &out) {
  if (!file_exists(path)) {  // absent: write shipped defaults once
    out = default_agents(repo_dir);
    save_agents(path, out);
    return true;
  }
  if (load_agents(path, out)) return true;
  out = default_agents(repo_dir);  // malformed: defaults in memory, file preserved
  return false;
}
