#include "run_manager.h"
#include "fsutil.h"
#include "json.h"
#include <cerrno>
#include <cctype>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <sstream>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

std::vector<std::string> split_args(const std::string &s) {
  std::istringstream in(s);
  std::vector<std::string> v;
  std::string t;
  while (in >> t) v.push_back(t);
  return v;
}

std::vector<int> scan_llama_processes(const std::string &proc_root) {
  std::vector<int> out;
  DIR *dir = opendir(proc_root.c_str());
  if (!dir) return out;
  int self = (int)getpid();
  while (dirent *e = readdir(dir)) {
    const char *n = e->d_name;
    if (!std::isdigit((unsigned char)n[0])) continue;
    int pid = std::atoi(n);
    if (pid <= 0 || pid == self) continue;
    bool ok = false;
    std::string comm = slurp(proc_root + "/" + n + "/comm", ok);
    if (ok && comm.rfind("llama-", 0) == 0) out.push_back(pid);
  }
  closedir(dir);
  return out;
}

bool parse_llama_bench_json(const std::string &text, double &tg_ts, double &pp_ts) {
  tg_ts = -1;
  pp_ts = -1;
  JsonValue v;
  if (!json_parse(text, v) || v.type != JsonValue::Type::Arr) return false;
  for (const auto &row : v.arr) {
    const JsonValue *ng = row.find("n_gen"), *np = row.find("n_prompt"), *ts = row.find("avg_ts");
    if (!ts) continue;
    if (ng && ng->i > 0) tg_ts = ts->as_number();
    else if (np && np->i > 0) pp_ts = ts->as_number();
  }
  return tg_ts >= 0 || pp_ts >= 0;
}

std::string run_record_to_jsonl(const RunRecord &r) {
  std::map<std::string, JsonValue> m;
  m["utc_start"] = JsonValue::make_str(r.utc_start);
  m["utc_end"] = JsonValue::make_str(r.utc_end);
  m["label"] = JsonValue::make_str(r.label);
  m["model_path"] = JsonValue::make_str(r.model_path);
  m["binary"] = JsonValue::make_str(r.binary);
  m["args"] = JsonValue::make_str(r.args);
  m["log_path"] = JsonValue::make_str(r.log_path);
  m["output_json_path"] = JsonValue::make_str(r.output_json_path);
  m["exit_code"] = JsonValue::make_int(r.exit_code);
  if (!r.note.empty()) m["note"] = JsonValue::make_str(r.note);
  if (r.tg_ts >= 0) m["tg_ts"] = JsonValue::make_double(r.tg_ts);
  if (r.pp_ts >= 0) m["pp_ts"] = JsonValue::make_double(r.pp_ts);
  return json_stringify(JsonValue::make_obj(std::move(m)));
}

static bool append_line(const std::string &path, const std::string &line) {
  FILE *f = std::fopen(path.c_str(), "a");
  if (!f) return false;
  std::fprintf(f, "%s\n", line.c_str());
  std::fclose(f);
  return true;
}

RunManager::RunManager(std::string out_dir, std::string bin_dir)
    : out_dir_(std::move(out_dir)), bin_dir_(std::move(bin_dir)) {
  ensure_dir(out_dir_);
}

RunManager::~RunManager() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    stop_all_ = true;
    if (child_pid_ > 0) detach_req_ = true;  // never kill a child implicitly
  }
  if (th_.joinable()) th_.join();
}

void RunManager::enqueue(QueuedRun r) {
  std::lock_guard<std::mutex> lk(mu_);
  queue_.push_back(std::move(r));
}

bool RunManager::remove_queued(size_t idx) {
  std::lock_guard<std::mutex> lk(mu_);
  if (idx >= queue_.size()) return false;
  queue_.erase(queue_.begin() + (long)idx);
  return true;
}

bool RunManager::start() {
  std::lock_guard<std::mutex> lk(mu_);
  if (th_.joinable()) {
    if (thread_done_) th_.join();
    else return false;
  }
  if (queue_.empty()) return false;
  stop_all_ = false;
  term_req_ = false;
  detach_req_ = false;
  thread_done_ = false;
  th_ = std::thread(&RunManager::run_loop, this);
  return true;
}

void RunManager::stop() {
  std::lock_guard<std::mutex> lk(mu_);
  term_req_ = true;
  stop_all_ = true;
}

void RunManager::detach() {
  std::lock_guard<std::mutex> lk(mu_);
  detach_req_ = true;
  stop_all_ = true;
}

RunStatus RunManager::status() {
  std::lock_guard<std::mutex> lk(mu_);
  RunStatus s = status_;
  if (s.active) {
    s.elapsed_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - arm_t0_).count();
    s.eta_s = -1;
    if (!s.rep_seconds.empty()) {
      double mean = 0;
      for (double x : s.rep_seconds) mean += x;
      mean /= s.rep_seconds.size();
      long long remaining = (long long)s.arm_count - s.arm_index;
      for (const auto &q : queue_) remaining += q.preset.repeats;
      s.eta_s = mean * (double)remaining;
    }
  }
  return s;
}

std::vector<QueuedRun> RunManager::queue_snapshot() {
  std::lock_guard<std::mutex> lk(mu_);
  return queue_;
}

std::vector<std::string> RunManager::tail_snapshot() {
  std::lock_guard<std::mutex> lk(mu_);
  return std::vector<std::string>(tail_.begin(), tail_.end());
}

std::vector<RunRecord> RunManager::records() {
  std::lock_guard<std::mutex> lk(mu_);
  return records_;
}

int RunManager::preflight(std::vector<int> &offending) {
  offending = scan_llama_processes("/proc");
  return (int)offending.size();
}

int RunManager::active_pid() {
  std::lock_guard<std::mutex> lk(mu_);
  return (int)child_pid_;
}

void RunManager::push_tail(const std::string &line) {
  std::lock_guard<std::mutex> lk(mu_);
  tail_.push_back(line);
  while (tail_.size() > 200) tail_.pop_front();
}

void RunManager::run_loop() {
  for (;;) {
    QueuedRun qr;
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (stop_all_ || queue_.empty()) break;
      qr = queue_.front();
      queue_.erase(queue_.begin());
      status_.active = true;
      status_.label = qr.preset.name;
      status_.arm_count = qr.preset.repeats;
      status_.arm_index = 0;
      status_.rep_seconds.clear();
      status_.phase = "running";
    }
    for (int rep = 1; rep <= qr.preset.repeats; ++rep) {
      {
        std::lock_guard<std::mutex> lk(mu_);
        if (stop_all_) break;
        status_.arm_index = rep;
        arm_t0_ = std::chrono::steady_clock::now();
      }
      RunRecord rec;
      rec.label = qr.preset.name;
      rec.model_path = qr.preset.model_path;
      rec.binary = qr.preset.binary;
      rec.args = qr.preset.args;
      rec.utc_start = utc_now();
      int rc = spawn_and_wait(qr.preset, rep, rec);
      double secs;
      {
        std::lock_guard<std::mutex> lk(mu_);
        secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - arm_t0_).count();
      }
      rec.utc_end = utc_now();
      rec.exit_code = rc;
      bool ok = false;
      std::string out_text = slurp(rec.output_json_path, ok);
      if (ok) {
        double tg, pp;
        if (parse_llama_bench_json(out_text, tg, pp)) {
          rec.tg_ts = tg;
          rec.pp_ts = pp;
        }
      }
      std::string line = run_record_to_jsonl(rec);
      if (!append_line(out_dir_ + "/runs.jsonl", line) && rec.note.empty())
        rec.note = "jsonl-append-failed";
      {
        std::lock_guard<std::mutex> lk(mu_);
        records_.push_back(rec);
        status_.rep_seconds.push_back(secs);
      }
      if (stop_all_) break;
    }
    std::lock_guard<std::mutex> lk(mu_);
    status_.active = false;
    status_.phase = "idle";
    if (stop_all_) break;
  }
  std::lock_guard<std::mutex> lk(mu_);
  status_.active = false;
  status_.phase = "idle";
  thread_done_ = true;
}

int RunManager::spawn_and_wait(const Preset &p, int rep, RunRecord &rec) {
  std::string base = out_dir_ + "/" + utc_stamp() + "-" + sanitize_label(p.name) + "-r" +
                     std::to_string(rep) + "-" + std::to_string(++spawn_seq_);
  rec.log_path = base + ".log";          // stderr
  rec.output_json_path = base + ".out";  // stdout
  int fd_out = open(rec.output_json_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  int fd_err = open(rec.log_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd_out < 0 || fd_err < 0) {
    rec.note = "open-failed";
    if (fd_out >= 0) close(fd_out);
    if (fd_err >= 0) close(fd_err);
    return -1;
  }
  std::string bin = p.binary.find('/') != std::string::npos ? p.binary : bin_dir_ + "/" + p.binary;
  std::vector<std::string> parts = split_args(p.args);
  std::vector<char *> av;
  av.push_back((char *)bin.c_str());
  for (auto &s : parts) av.push_back((char *)s.c_str());
  av.push_back(nullptr);
  pid_t pid = fork();
  if (pid < 0) {
    rec.note = "fork-failed";
    close(fd_out);
    close(fd_err);
    return -1;
  }
  if (pid == 0) {
    setsid();
    dup2(fd_out, STDOUT_FILENO);
    dup2(fd_err, STDERR_FILENO);
    close(fd_out);
    close(fd_err);
    execvp(bin.c_str(), av.data());
    const char msg[] = "bench-tui: execvp failed\n";
    (void)!write(STDERR_FILENO, msg, sizeof(msg) - 1);
    _exit(127);
  }
  close(fd_out);
  close(fd_err);
  {
    std::lock_guard<std::mutex> lk(mu_);
    child_pid_ = pid;
  }
  off_t tail_off = 0;
  bool term_sent = false;
  std::chrono::steady_clock::time_point kill_at{};
  int st = 0;
  for (;;) {
    int fd = open(rec.log_path.c_str(), O_RDONLY);
    if (fd >= 0) {
      if (lseek(fd, tail_off, SEEK_SET) == tail_off) {
        char buf[8192];
        ssize_t n = read(fd, buf, sizeof buf);
        if (n > 0) {
          tail_off += n;
          std::string chunk(buf, (size_t)n);
          for (auto &c : chunk)
            if (c == '\r') c = '\n';
          tail_partial_ += chunk;
          size_t pos;
          while ((pos = tail_partial_.find('\n')) != std::string::npos) {
            push_tail(tail_partial_.substr(0, pos));
            tail_partial_.erase(0, pos + 1);
          }
        }
      }
      close(fd);
    }
    pid_t r = waitpid(pid, &st, WNOHANG);
    if (r == pid) break;
    if (r < 0 && errno == ECHILD) break;  // child already gone
    bool term = false, det = false;
    {
      std::lock_guard<std::mutex> lk(mu_);
      term = term_req_;
      det = detach_req_;
    }
    if (det) {
      std::lock_guard<std::mutex> lk(mu_);
      rec.note = "detached";
      child_pid_ = -1;
      return -1;  // child keeps running; the record has no exit code
    }
    if (term && !term_sent) {
      kill(-pid, SIGTERM);  // the child is its own process-group leader (setsid)
      term_sent = true;
      kill_at = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    }
    if (term_sent && std::chrono::steady_clock::now() > kill_at) {
      kill(-pid, SIGKILL);
      kill_at = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  {
    std::lock_guard<std::mutex> lk(mu_);
    child_pid_ = -1;
  }
  if (term_sent && rec.note.empty()) rec.note = "stopped";
  if (WIFEXITED(st)) return WEXITSTATUS(st);
  if (WIFSIGNALED(st)) return -WTERMSIG(st);
  return -1;
}
