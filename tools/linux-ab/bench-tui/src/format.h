#pragma once
// Human-readable formatters for the UI. Pure functions.
#include <string>

std::string format_gib(double mb);       // 512 -> "512 MiB"; 2048 -> "2.00 GiB"
std::string format_mbs(double kb_s);     // KiB/s input, human unit output
std::string format_duration(double s);   // 754 -> "12:34"; 3725 -> "1:02:05"; <0 -> "--:--"
std::string format_pct(float v);         // 42.4 -> "42%"
std::string format_ts(double v);         // 44.472772 -> "44.47"
