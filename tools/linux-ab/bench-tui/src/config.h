#pragma once
// Preset/agent config: JSON files, hand-editable, loaded at start, saved on edit.
#include "json.h"
#include <string>
#include <vector>

struct Preset {
  std::string name, model_path, binary, args;
  int repeats = 1;
};

struct AgentSpec {
  std::string name, command, workdir;
};

// Loaders never throw. Missing file -> defaults, returns true.
// Malformed file -> defaults, returns false; the file is left untouched.
bool load_presets(const std::string &path, std::vector<Preset> &out);
bool save_presets(const std::string &path, const std::vector<Preset> &v);
bool load_agents(const std::string &path, std::vector<AgentSpec> &out);
bool save_agents(const std::string &path, const std::vector<AgentSpec> &v);
std::vector<AgentSpec> default_agents(const std::string &repo_dir);
// Startup helpers: write default config ONLY when the file is absent. An
// existing file is never rewritten, malformed or not. Both return false when
// the file existed but failed to parse (defaults in memory, original preserved).
bool ensure_presets_config(const std::string &path, std::vector<Preset> &out);
bool ensure_agents_config(const std::string &path, const std::string &repo_dir,
                          std::vector<AgentSpec> &out);
