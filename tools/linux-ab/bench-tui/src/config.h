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
