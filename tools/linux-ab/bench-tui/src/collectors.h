#pragma once
// Telemetry collectors: /proc parsers, NVML/nvidia-smi GPU sampling, and the
// background CollectorRunner feeding CollectorState. All strictly read-only.
#include <atomic>
#include <chrono>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

struct CpuSample {
  std::vector<float> per_core;  // busy % per logical core
  float load1 = 0.0f;
};
struct MemSample {
  long total_mb = 0, avail_mb = 0, swap_total_mb = 0, swap_free_mb = 0;
};
struct DiskSample {
  std::string device;
  double read_kbs = 0.0, write_kbs = 0.0;
};
struct GpuSample {
  bool available = false;
  unsigned util_pct = 0, temp_c = 0, power_w = 0;
  unsigned long long vram_used_mb = 0, vram_total_mb = 0;
  std::string source;  // "nvml" or "nvidia-smi"
};

// Pure parsers (unit-tested on fixtures):
bool parse_meminfo(const std::string &text, MemSample &out);
bool parse_cpu_times(const std::string &text, std::vector<std::pair<long long, long long>> &out);
bool parse_diskstats(const std::string &text,
                     std::map<std::string, std::pair<unsigned long long, unsigned long long>> &out);
bool parse_nvidia_smi_csv(const std::string &line, GpuSample &out);

// Live samplers; deltas are computed against the previous call:
class CpuSampler {
public:
  CpuSample sample();
private:
  std::vector<std::pair<long long, long long>> prev_;
};
class DiskSampler {
public:
  std::vector<DiskSample> sample();
private:
  std::map<std::string, std::pair<unsigned long long, unsigned long long>> prev_;
  std::chrono::steady_clock::time_point prev_t_{};
};
class GpuSampler {
public:
  GpuSampler();
  ~GpuSampler();
  GpuSample sample();
private:
  void *lib_ = nullptr;
  bool nvml_ok_ = false, smi_available_ = false;
  void *dev_ = nullptr;
  void *p_shutdown_ = nullptr, *p_util_ = nullptr, *p_mem_ = nullptr, *p_temp_ = nullptr, *p_power_ = nullptr;
};
MemSample sample_mem();

inline void push_hist(std::deque<float> &h, float v) {
  h.push_back(v);
  while (h.size() > 120) h.pop_front();
}

struct CollectorState {
  std::mutex mu;
  CpuSample cpu;
  MemSample mem;
  std::vector<DiskSample> disks;
  GpuSample gpu;
  std::deque<float> gpu_util_hist, vram_hist, ram_hist, disk_hist;
};

// Plain copy of CollectorState contents (the mutex itself is not copied).
struct CollectorSnapshot {
  CpuSample cpu;
  MemSample mem;
  std::vector<DiskSample> disks;
  GpuSample gpu;
  std::deque<float> gpu_util_hist, vram_hist, ram_hist, disk_hist;
};
CollectorSnapshot snapshot_from(CollectorState &st);

class CollectorRunner {
public:
  explicit CollectorRunner(CollectorState &st);
  ~CollectorRunner();  // stop + join
  void start();
  void stop();
  void set_interval_ms(int ms);  // quiet-mode control (1500 idle, 5000 during runs)
private:
  void loop();
  CollectorState &st_;
  CpuSampler cpu_;
  DiskSampler disk_;
  GpuSampler gpu_;
  std::atomic<int> interval_ms_{1500};
  std::atomic<bool> stop_{false};
  std::thread th_;
};
