#pragma once
// Run manager: owns the preset queue and every spawned child. Children are
// launched with stdout/stderr redirected to files under out_dir so detach is
// safe (no pipes, no SIGPIPE). Records append to out_dir/runs.jsonl.
#include "config.h"
#include <atomic>
#include <chrono>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <sys/types.h>  // pid_t

struct QueuedRun {
  Preset preset;
};

struct RunRecord {
  std::string utc_start, utc_end, label, model_path, binary, args;
  std::string log_path, output_json_path, note;  // note: "", "stopped", "detached", spawn errors
  int exit_code = -1;                            // negative: -signal; -1 also used for detached
  double tg_ts = -1, pp_ts = -1;                 // <0 = absent (null in runs.jsonl)
};

struct RunStatus {
  bool active = false;
  std::string label, phase;  // phase: "idle", "running"
  int arm_index = 0, arm_count = 0;
  double elapsed_s = 0, eta_s = -1;
  std::vector<double> rep_seconds;
};

class RunManager {
public:
  RunManager(std::string out_dir, std::string bin_dir);
  ~RunManager();  // joins the worker; detaches an active child, but completes an in-flight stop()
  void enqueue(QueuedRun r);
  bool remove_queued(size_t idx);
  bool start();   // false when already running or the queue is empty; re-joins a finished worker
  void stop();    // SIGTERM -> 5 s grace -> SIGKILL
  void detach();  // stop managing; the child keeps running
  RunStatus status();
  std::vector<QueuedRun> queue_snapshot();
  std::vector<std::string> tail_snapshot();  // ring buffer of child stderr lines (last 200)
  std::vector<RunRecord> records();
  int preflight(std::vector<int> &offending);  // count of llama-* processes; 0 = clear
  int active_pid();                            // 0 when no child; test support
private:
  void run_loop();
  int spawn_and_wait(const Preset &p, int rep, RunRecord &rec);
  void push_tail(const std::string &line);
  std::string out_dir_, bin_dir_;
  std::mutex mu_;
  std::vector<QueuedRun> queue_;
  std::vector<RunRecord> records_;
  std::deque<std::string> tail_;
  std::string tail_partial_;
  RunStatus status_;
  std::chrono::steady_clock::time_point arm_t0_{};
  std::thread th_;
  bool thread_done_ = true;
  std::atomic<bool> stop_all_ = false, term_req_ = false, detach_req_ = false;
  pid_t child_pid_ = -1;
  unsigned spawn_seq_ = 0;
};

std::vector<std::string> split_args(const std::string &s);  // whitespace split; no quoting by design
std::vector<int> scan_llama_processes(const std::string &proc_root);
bool parse_llama_bench_json(const std::string &text, double &tg_ts, double &pp_ts);
std::string run_record_to_jsonl(const RunRecord &r);
