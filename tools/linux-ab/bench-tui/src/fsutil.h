#pragma once
// Small POSIX filesystem helpers shared by config, run manager, and UI.
#include <string>
#include <vector>

bool file_exists(const std::string &path);
bool ensure_dir(const std::string &path);                     // mkdir -p semantics
std::string slurp(const std::string &path, bool &ok);         // whole file; ok=false when unreadable
bool write_file(const std::string &path, const std::string &content);
std::vector<std::string> scan_gguf(const std::string &dir);   // *.gguf paths, sorted; empty when dir missing
std::string sanitize_label(std::string s);                    // filename-safe run label
std::string utc_now();    // 2026-08-14T05:33:04Z
std::string utc_stamp();  // 20260814T053304Z for filenames
