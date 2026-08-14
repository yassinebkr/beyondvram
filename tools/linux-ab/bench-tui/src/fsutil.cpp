#include "fsutil.h"
#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstdio>
#include <ctime>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

bool file_exists(const std::string &path) { return access(path.c_str(), F_OK) == 0; }

bool ensure_dir(const std::string &path) {
  if (path.empty()) return false;
  std::string cur;
  for (char c : path) {
    cur += c;
    if (c == '/' && cur.size() > 1 && mkdir(cur.c_str(), 0755) != 0 && errno != EEXIST)
      return false;
  }
  return mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
}

std::string slurp(const std::string &path, bool &ok) {
  ok = false;
  FILE *f = std::fopen(path.c_str(), "rb");
  if (!f) return "";
  std::string out;
  char buf[8192];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) out.append(buf, n);
  std::fclose(f);
  ok = true;
  return out;
}

bool write_file(const std::string &path, const std::string &content) {
  FILE *f = std::fopen(path.c_str(), "wb");
  if (!f) return false;
  bool ok = std::fwrite(content.data(), 1, content.size(), f) == content.size();
  std::fclose(f);
  return ok;
}

std::vector<std::string> scan_gguf(const std::string &dir) {
  std::vector<std::string> out;
  DIR *d = opendir(dir.c_str());
  if (!d) return out;
  while (dirent *e = readdir(d)) {
    std::string n = e->d_name;
    if (n.size() > 5 && n.compare(n.size() - 5, 5, ".gguf") == 0)
      out.push_back(dir + "/" + n);
  }
  closedir(d);
  std::sort(out.begin(), out.end());
  return out;
}

std::string sanitize_label(std::string s) {
  for (auto &c : s)
    if (!std::isalnum((unsigned char)c) && c != '-' && c != '_') c = '-';
  return s.empty() ? "run" : s;
}

static std::string utc_fmt(const char *fmt) {
  char b[32];
  std::time_t t = std::time(nullptr);
  std::tm tm;
  gmtime_r(&t, &tm);
  std::strftime(b, sizeof b, fmt, &tm);
  return b;
}
std::string utc_now() { return utc_fmt("%Y-%m-%dT%H:%M:%SZ"); }
std::string utc_stamp() { return utc_fmt("%Y%m%dT%H%M%SZ"); }
