#include "collectors.h"
#include "fsutil.h"
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <sstream>

bool parse_meminfo(const std::string &text, MemSample &out) {
  bool got_total = false, got_avail = false;
  std::istringstream in(text);
  std::string line;
  auto kv_kb = [&line](const char *key, long &mb) {
    size_t k = line.find(':');
    if (k == std::string::npos || line.compare(0, k, key) != 0) return false;
    mb = std::atol(line.c_str() + k + 1) / 1024;
    return true;
  };
  while (std::getline(in, line)) {
    if (kv_kb("MemTotal", out.total_mb)) got_total = true;
    else if (kv_kb("MemAvailable", out.avail_mb)) got_avail = true;
    else if (kv_kb("SwapTotal", out.swap_total_mb)) {}
    else if (kv_kb("SwapFree", out.swap_free_mb)) {}
  }
  return got_total && got_avail;
}

bool parse_cpu_times(const std::string &text, std::vector<std::pair<long long, long long>> &out) {
  out.clear();
  std::istringstream in(text);
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind("cpu", 0) != 0) continue;
    if (line.size() < 4 || !std::isdigit((unsigned char)line[3])) continue;  // skip aggregate "cpu "
    std::istringstream ls(line);
    std::string tag;
    long long user = 0, nice = 0, system = 0, idle = 0, iowait = 0, irq = 0, softirq = 0;
    ls >> tag >> user >> nice >> system >> idle >> iowait >> irq >> softirq;
    long long busy = user + nice + system + irq + softirq;
    long long total = busy + idle + iowait;
    if (total <= 0) return false;
    out.push_back({busy, total});
  }
  return !out.empty();
}

bool parse_diskstats(const std::string &text,
                     std::map<std::string, std::pair<unsigned long long, unsigned long long>> &out) {
  out.clear();
  std::istringstream in(text);
  std::string line;
  while (std::getline(in, line)) {
    std::istringstream ls(line);
    std::string major, minor, name;
    if (!(ls >> major >> minor >> name)) continue;
    if (name.rfind("loop", 0) == 0 || name.rfind("ram", 0) == 0 || name.rfind("sr", 0) == 0) continue;
    if (name.rfind("nvme", 0) != 0 && name.rfind("sd", 0) != 0 && name.rfind("vd", 0) != 0 &&
        name.rfind("mmcblk", 0) != 0)
      continue;
    unsigned long long rd = 0, rm = 0, sr = 0, tr = 0, wr = 0, wm = 0, sw = 0;
    if (!(ls >> rd >> rm >> sr >> tr >> wr >> wm >> sw)) continue;
    out[name] = {sr, sw};  // 512-byte sectors read / written
  }
  return !out.empty();
}

bool parse_nvidia_smi_csv(const std::string &line, GpuSample &out) {
  std::vector<std::string> f;
  std::string cur;
  std::istringstream in(line);
  while (std::getline(in, cur, ',')) f.push_back(cur);
  if (f.size() < 5) return false;
  auto trim = [](std::string s) {
    size_t a = s.find_first_not_of(" \t");
    size_t b = s.find_last_not_of(" \t");
    return a == std::string::npos ? std::string() : s.substr(a, b - a + 1);
  };
  out.util_pct = (unsigned)std::atol(trim(f[0]).c_str());
  out.vram_used_mb = std::strtoull(trim(f[1]).c_str(), nullptr, 10);
  out.vram_total_mb = std::strtoull(trim(f[2]).c_str(), nullptr, 10);
  out.temp_c = (unsigned)std::atol(trim(f[3]).c_str());
  out.power_w = (unsigned)(std::atof(trim(f[4]).c_str()) + 0.5);
  out.available = true;
  out.source = "nvidia-smi";
  return true;
}

CpuSample CpuSampler::sample() {
  CpuSample out;
  bool ok = false;
  std::string text = slurp("/proc/stat", ok);
  std::vector<std::pair<long long, long long>> cur;
  if (ok && parse_cpu_times(text, cur)) {
    if (prev_.size() == cur.size()) {
      for (size_t k = 0; k < cur.size(); ++k) {
        long long db = cur[k].first - prev_[k].first, dt = cur[k].second - prev_[k].second;
        out.per_core.push_back(dt > 0 ? (float)(100.0 * db / dt) : 0.0f);
      }
    }
    prev_ = std::move(cur);
  }
  bool ok2 = false;
  std::string la = slurp("/proc/loadavg", ok2);
  if (ok2) out.load1 = (float)std::atof(la.c_str());
  return out;
}

std::vector<DiskSample> DiskSampler::sample() {
  std::vector<DiskSample> out;
  bool ok = false;
  std::string text = slurp("/proc/diskstats", ok);
  std::map<std::string, std::pair<unsigned long long, unsigned long long>> cur;
  if (!ok || !parse_diskstats(text, cur)) return out;
  auto now = std::chrono::steady_clock::now();
  double dt = std::chrono::duration<double>(now - prev_t_).count();
  if (!prev_.empty() && dt > 0) {
    for (const auto &kv : cur) {
      auto it = prev_.find(kv.first);
      if (it == prev_.end()) continue;
      DiskSample d;
      d.device = kv.first;
      d.read_kbs = (kv.second.first - it->second.first) / 2.0 / dt;
      d.write_kbs = (kv.second.second - it->second.second) / 2.0 / dt;
      out.push_back(d);
    }
    std::sort(out.begin(), out.end(), [](const DiskSample &a, const DiskSample &b) {
      return a.read_kbs + a.write_kbs > b.read_kbs + b.write_kbs;
    });
  }
  prev_ = std::move(cur);
  prev_t_ = now;
  return out;
}

MemSample sample_mem() {
  MemSample m;
  bool ok = false;
  std::string t = slurp("/proc/meminfo", ok);
  if (ok) parse_meminfo(t, m);
  return m;
}

// NVML is dlopen'd at runtime so bench-tui builds and runs without the driver.
// Only the stable, versioned entry points used here are declared.
namespace {
struct nvmlUtilization_t { unsigned int gpu; unsigned int memory; };
struct nvmlMemory_t { unsigned long long total; unsigned long long free; unsigned long long used; };
typedef int (*nvmlInit_t)();
typedef int (*nvmlShutdown_t)();
typedef int (*nvmlGetHandle_t)(unsigned int, void **);
typedef int (*nvmlUtil_t)(void *, nvmlUtilization_t *);
typedef int (*nvmlMem_t)(void *, nvmlMemory_t *);
typedef int (*nvmlTemp_t)(void *, int, unsigned int *);
typedef int (*nvmlPower_t)(void *, unsigned int *);
}  // namespace

GpuSampler::GpuSampler() {
  const char *cands[] = {"libnvidia-ml.so.1", "/usr/lib/wsl/lib/libnvidia-ml.so.1"};
  for (const char *c : cands) {
    lib_ = dlopen(c, RTLD_NOW | RTLD_LOCAL);
    if (lib_) break;
  }
  if (lib_) {
    nvmlInit_t p_init = (nvmlInit_t)dlsym(lib_, "nvmlInit_v2");
    p_shutdown_ = dlsym(lib_, "nvmlShutdown");
    p_util_ = dlsym(lib_, "nvmlDeviceGetUtilizationRates");
    p_mem_ = dlsym(lib_, "nvmlDeviceGetMemoryInfo");
    p_temp_ = dlsym(lib_, "nvmlDeviceGetTemperature");
    p_power_ = dlsym(lib_, "nvmlDeviceGetPowerUsage");
    nvmlGetHandle_t p_handle = (nvmlGetHandle_t)dlsym(lib_, "nvmlDeviceGetHandleByIndex_v2");
    if (p_init && p_shutdown_ && p_util_ && p_mem_ && p_temp_ && p_power_ && p_handle &&
        p_init() == 0 && p_handle(0, &dev_) == 0)
      nvml_ok_ = true;
  }
  if (!nvml_ok_)
    smi_available_ = (std::system("command -v nvidia-smi >/dev/null 2>&1") == 0);
}

GpuSampler::~GpuSampler() {
  if (nvml_ok_) ((nvmlShutdown_t)p_shutdown_)();
  if (lib_) dlclose(lib_);
}

GpuSample GpuSampler::sample() {
  GpuSample s;
  if (nvml_ok_) {
    nvmlUtilization_t u{};
    nvmlMemory_t m{};
    unsigned int t = 0, pw = 0;
    if (((nvmlUtil_t)p_util_)(dev_, &u) == 0) s.util_pct = u.gpu;
    if (((nvmlMem_t)p_mem_)(dev_, &m) == 0) {
      s.vram_total_mb = m.total >> 20;
      s.vram_used_mb = m.used >> 20;
    }
    if (((nvmlTemp_t)p_temp_)(dev_, 0, &t) == 0) s.temp_c = t;  // 0 == NVML_TEMPERATURE_GPU
    if (((nvmlPower_t)p_power_)(dev_, &pw) == 0) s.power_w = pw / 1000;  // mW -> W
    s.available = true;
    s.source = "nvml";
    return s;
  }
  if (smi_available_) {
    FILE *f = popen(
        "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw "
        "--format=csv,noheader,nounits 2>/dev/null",
        "r");
    if (f) {
      char buf[256];
      if (fgets(buf, sizeof buf, f)) parse_nvidia_smi_csv(buf, s);
      pclose(f);
    }
  }
  return s;
}

CollectorSnapshot snapshot_from(CollectorState &st) {
  std::lock_guard<std::mutex> lk(st.mu);
  CollectorSnapshot s;
  s.cpu = st.cpu;
  s.mem = st.mem;
  s.disks = st.disks;
  s.gpu = st.gpu;
  s.gpu_util_hist = st.gpu_util_hist;
  s.vram_hist = st.vram_hist;
  s.ram_hist = st.ram_hist;
  s.disk_hist = st.disk_hist;
  return s;
}

CollectorRunner::CollectorRunner(CollectorState &st) : st_(st) {}
CollectorRunner::~CollectorRunner() { stop(); }

void CollectorRunner::start() {
  stop_ = false;
  th_ = std::thread(&CollectorRunner::loop, this);
}

void CollectorRunner::stop() {
  stop_ = true;
  if (th_.joinable()) th_.join();
}

void CollectorRunner::set_interval_ms(int ms) { interval_ms_ = ms; }

void CollectorRunner::loop() {
  while (!stop_) {
    CpuSample c = cpu_.sample();
    MemSample m = sample_mem();
    std::vector<DiskSample> d = disk_.sample();
    GpuSample g = gpu_.sample();
    {
      std::lock_guard<std::mutex> lk(st_.mu);
      st_.cpu = std::move(c);
      st_.mem = m;
      st_.disks = std::move(d);
      st_.gpu = g;
      if (g.available) {
        push_hist(st_.gpu_util_hist, (float)g.util_pct);
        if (g.vram_total_mb) push_hist(st_.vram_hist, 100.0f * g.vram_used_mb / g.vram_total_mb);
      }
      if (m.total_mb > 0)
        push_hist(st_.ram_hist, 100.0f * (m.total_mb - m.avail_mb) / m.total_mb);
      double kbs = 0;
      for (const auto &ds : st_.disks) kbs += ds.read_kbs + ds.write_kbs;
      push_hist(st_.disk_hist, (float)(kbs / 1024.0));
    }
    int waited = 0;
    while (!stop_ && waited < interval_ms_) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      waited += 50;
    }
  }
}
