#include "format.h"
#include <cstdio>

std::string format_gib(double mb) {
  char b[32];
  if (mb >= 1024.0) std::snprintf(b, sizeof b, "%.2f GiB", mb / 1024.0);
  else std::snprintf(b, sizeof b, "%.0f MiB", mb);
  return b;
}

std::string format_mbs(double kb_s) {
  char b[32];
  if (kb_s >= 1048576.0) std::snprintf(b, sizeof b, "%.2f GiB/s", kb_s / 1048576.0);
  else if (kb_s >= 1024.0) std::snprintf(b, sizeof b, "%.1f MiB/s", kb_s / 1024.0);
  else std::snprintf(b, sizeof b, "%.0f KiB/s", kb_s);
  return b;
}

std::string format_duration(double s) {
  if (s < 0) return "--:--";
  long v = (long)(s + 0.5);
  char b[32];
  if (v >= 3600) std::snprintf(b, sizeof b, "%ld:%02ld:%02ld", v / 3600, (v % 3600) / 60, v % 60);
  else std::snprintf(b, sizeof b, "%ld:%02ld", v / 60, v % 60);
  return b;
}

std::string format_pct(float v) {
  char b[16];
  std::snprintf(b, sizeof b, "%.0f%%", (double)v);
  return b;
}

std::string format_ts(double v) {
  char b[16];
  std::snprintf(b, sizeof b, "%.2f", v);
  return b;
}
