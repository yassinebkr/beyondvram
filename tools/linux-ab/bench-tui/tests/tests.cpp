// bench-tui unit tests. Single-file CHECK runner; TEST blocks self-register.
// Exit code is 0 when nothing failed, 1 otherwise.
#include "collectors.h"
#include "config.h"
#include "fsutil.h"
#include "json.h"
#include "run_manager.h"
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <map>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>
#include <vector>

static int g_failures = 0;

#define CHECK(cond)                                                          \
  do {                                                                       \
    if (!(cond)) {                                                           \
      std::printf("  CHECK failed %s:%d: %s\n", __FILE__, __LINE__, #cond);  \
      ++g_failures;                                                          \
    }                                                                        \
  } while (0)

using TestFn = void (*)();
static std::vector<std::pair<const char *, TestFn>> &registry() {
  static std::vector<std::pair<const char *, TestFn>> r;
  return r;
}
struct AddTest {
  AddTest(const char *name, TestFn fn) { registry().push_back({name, fn}); }
};
#define TEST(name)                                    \
  static void name();                                 \
  static AddTest add_##name(#name, name);             \
  static void name()

TEST(smoke) { CHECK(1 + 1 == 2); }

int main() {
  for (auto &t : registry()) {
    int before = g_failures;
    t.second();
    std::printf("%s %s\n", g_failures == before ? "PASS" : "FAIL", t.first);
  }
  std::printf("%d failure(s)\n", g_failures);
  return g_failures ? 1 : 0;
}

TEST(json_flat_object) {
  JsonValue v;
  CHECK(json_parse("{\"name\":\"base\",\"repeats\":3,\"on\":true}", v));
  CHECK(v.type == JsonValue::Type::Obj);
  CHECK(v.find("name")->s == "base");
  CHECK(v.find("repeats")->i == 3);
  CHECK(v.find("on")->b == true);
  CHECK(v.find("missing") == nullptr);
}

TEST(json_number_kinds) {
  JsonValue v;
  CHECK(json_parse("[44, 44.472772, -3, 1e3]", v));
  CHECK(v.arr[0].type == JsonValue::Type::Int);
  CHECK(v.arr[1].type == JsonValue::Type::Double);
  CHECK(v.arr[1].as_number() > 44.47 && v.arr[1].as_number() < 44.48);
  CHECK(v.arr[2].i == -3);
  CHECK(v.arr[3].as_number() == 1000.0);
}

TEST(json_escapes) {
  JsonValue v;
  CHECK(json_parse("\"a\\nb\\t\\\"c\\\"\\u00e9\"", v));
  CHECK(v.s == "a\nb\t\"c\"\xc3\xa9");  // U+00E9 as UTF-8
}

TEST(json_rejects_malformed) {
  JsonValue v;
  CHECK(!json_parse("{\"a\":1", v));
  CHECK(!json_parse("[1,2] trailing", v));
  CHECK(!json_parse("\"unterminated", v));
  CHECK(!json_parse("", v));
}

TEST(json_roundtrip) {
  std::map<std::string, JsonValue> m;
  m["args"] = JsonValue::make_str("-ngl 48 --n-cpu-moe 33");
  m["repeats"] = JsonValue::make_int(5);
  m["nested"] = JsonValue::make_arr({JsonValue::make_int(1), JsonValue::make_str("x")});
  JsonValue v;
  CHECK(json_parse(json_stringify(JsonValue::make_obj(m)), v));
  CHECK(v.find("args")->s == "-ngl 48 --n-cpu-moe 33");
  CHECK(v.find("nested")->arr[1].s == "x");
}

TEST(json_rejects_nonjson_numbers) {
  JsonValue v;
  CHECK(!json_parse("nan", v));
  CHECK(!json_parse("inf", v));
  CHECK(!json_parse("0x1F", v));
  CHECK(!json_parse("+5", v));
  CHECK(!json_parse(".5", v));
  CHECK(!json_parse("01", v));
  CHECK(!json_parse("1.", v));
  CHECK(!json_parse("{\"r\":01}", v));
  CHECK(!json_parse("{\"r\":nan}", v));
}

TEST(json_rejects_deep_nesting) {
  JsonValue v;
  std::string deep(200, '[');
  deep += std::string(200, ']');
  CHECK(!json_parse(deep, v));
  std::string shallow(50, '[');
  shallow += std::string(50, ']');
  CHECK(json_parse(shallow, v));
}

TEST(json_int64_range) {
  JsonValue v;
  CHECK(!json_parse("99999999999999999999999", v));
  CHECK(json_parse("-9223372036854775808", v));
  CHECK(v.type == JsonValue::Type::Int);
  CHECK(v.i == (-9223372036854775807LL - 1));
}

static std::string make_tmp_dir() {
  char t[] = "/tmp/bench-tui-test-XXXXXX";
  CHECK(mkdtemp(t) != nullptr);
  return t;
}

TEST(fsutil_dirs) {
  std::string base = make_tmp_dir();
  CHECK(ensure_dir(base + "/a/b/c"));
  CHECK(file_exists(base + "/a/b/c"));
  CHECK(!file_exists(base + "/a/b/nope"));
  CHECK(ensure_dir(base + "/a/b/c"));  // EEXIST stays true
}

TEST(fsutil_rw) {
  std::string base = make_tmp_dir();
  CHECK(write_file(base + "/x.txt", "hello\n"));
  bool ok = false;
  CHECK(slurp(base + "/x.txt", ok) == "hello\n" && ok);
  bool ok2 = true;
  slurp(base + "/missing", ok2);
  CHECK(!ok2);
}

TEST(fsutil_scan_gguf) {
  std::string base = make_tmp_dir();
  write_file(base + "/b.gguf", "x");
  write_file(base + "/a.gguf", "x");
  write_file(base + "/c.txt", "x");
  auto v = scan_gguf(base);
  CHECK(v.size() == 2);
  CHECK(v[0] == base + "/a.gguf" && v[1] == base + "/b.gguf");
  CHECK(scan_gguf(base + "/nope").empty());
}

TEST(fsutil_labels_time) {
  CHECK(sanitize_label("a b/c.d") == "a-b-c-d");
  CHECK(sanitize_label("") == "run");
  CHECK(utc_now().size() == 20 && utc_now().back() == 'Z');
  CHECK(utc_stamp().size() == 16);
}

TEST(fsutil_ensure_dir_rejects_file) {
  std::string base = make_tmp_dir();
  write_file(base + "/f", "x");
  CHECK(!ensure_dir(base + "/f"));
  CHECK(!ensure_dir(base + "/f/g"));
  CHECK(!ensure_dir(base + "/f/"));
  CHECK(ensure_dir(base + "/subdir"));
}

TEST(config_presets_roundtrip) {
  std::string base = make_tmp_dir();
  std::string path = base + "/presets.json";
  std::vector<Preset> in = {{"a", "/m/a.gguf", "llama-bench", "-ngl 0 -o json", 3},
                            {"b", "/m/b.gguf", "llama-server", "--port 8080", 1}};
  CHECK(save_presets(path, in));
  std::vector<Preset> out;
  CHECK(load_presets(path, out));
  CHECK(out.size() == 2);
  CHECK(out[0].name == "a" && out[0].repeats == 3 && out[0].args == "-ngl 0 -o json");
  CHECK(out[1].binary == "llama-server");
}

TEST(config_missing_and_malformed) {
  std::string base = make_tmp_dir();
  std::vector<Preset> out;
  CHECK(load_presets(base + "/nope.json", out) && out.empty());  // missing -> empty defaults
  std::string path = base + "/bad.json";
  write_file(path, "{not json");
  CHECK(!load_presets(path, out) && out.empty());  // malformed -> false
  bool ok = false;
  CHECK(slurp(path, ok) == "{not json");           // file preserved
}

TEST(config_agents) {
  auto d = default_agents("/repo");
  CHECK(d.size() == 3 && d[0].name == "kimi" && d[1].name == "codex");
  std::string base = make_tmp_dir();
  std::string path = base + "/agents.json";
  CHECK(save_agents(path, d));
  std::vector<AgentSpec> out;
  CHECK(load_agents(path, out));
  CHECK(out.size() == 3 && out[2].workdir == "/repo");
}

TEST(parse_meminfo_fixture) {
  MemSample m;
  CHECK(parse_meminfo("MemTotal:       65536 kB\nMemAvailable:    32768 kB\nSwapTotal:        2048 kB\nSwapFree:         1024 kB\n", m));
  CHECK(m.total_mb == 64 && m.avail_mb == 32 && m.swap_total_mb == 2 && m.swap_free_mb == 1);
}

TEST(parse_cpu_times_fixture) {
  std::vector<std::pair<long long, long long>> v;
  CHECK(parse_cpu_times("cpu  100 0 100 800 0 0 0 0 0 0\ncpu0 50 0 50 400 0 0 0 0 0 0\ncpu1 25 0 25 350 0 0 0 0 0 0\n", v));
  CHECK(v.size() == 2);
  CHECK(v[0].first == 100 && v[0].second == 500);
  CHECK(v[1].first == 50 && v[1].second == 400);
}

TEST(parse_diskstats_fixture) {
  std::map<std::string, std::pair<unsigned long long, unsigned long long>> m;
  CHECK(parse_diskstats(
      "   8       0 sda 100 0 2048 0 50 0 1024 0 0 0 0 0 0 0\n"
      "   7       0 loop0 5 0 10 0 0 0 0 0 0 0 0 0 0 0\n"
      " 259       0 nvme0n1 200 0 8192 0 100 0 4096 0 0 0 0 0 0 0\n", m));
  CHECK(m.count("loop0") == 0);
  CHECK(m["sda"].first == 2048 && m["sda"].second == 1024);
  CHECK(m["nvme0n1"].first == 8192 && m["nvme0n1"].second == 4096);
}

TEST(parse_smi_csv_fixture) {
  GpuSample g;
  CHECK(parse_nvidia_smi_csv("12, 2048, 8191, 55, 98.5\n", g));
  CHECK(g.available && g.util_pct == 12 && g.vram_used_mb == 2048 && g.vram_total_mb == 8191);
  CHECK(g.temp_c == 55 && g.power_w == 99);  // 98.5 rounds to 99
  CHECK(g.source == "nvidia-smi");
}

TEST(live_samplers_do_not_crash) {
  CpuSampler cs;
  cs.sample();
  usleep(60000);
  CpuSample c = cs.sample();
  CHECK(!c.per_core.empty());
  for (float f : c.per_core) CHECK(f >= 0.0f && f <= 100.0f);
  MemSample m = sample_mem();
  CHECK(m.total_mb > 0 && m.avail_mb >= 0);
  GpuSampler gs;
  GpuSample g1 = gs.sample();
  if (g1.available) CHECK(g1.vram_total_mb > 0);
}

TEST(collector_runner_fills_state) {
  CollectorState st;
  {
    CollectorRunner r(st);
    r.set_interval_ms(50);
    r.start();
    usleep(300000);
    r.stop();
  }
  CollectorSnapshot s = snapshot_from(st);
  CHECK(s.mem.total_mb > 0);
  CHECK(!s.ram_hist.empty() && s.ram_hist.size() <= 120);
}

TEST(push_hist_caps) {
  std::deque<float> h;
  for (int i = 0; i < 130; ++i) push_hist(h, (float)i);
  CHECK(h.size() == 120 && h.front() == 10.0f);
}

TEST(collector_runner_double_start) {
  CollectorState st;
  CollectorRunner r(st);
  r.set_interval_ms(50);
  r.start();
  r.start();  // second call while joinable must not crash
  usleep(150000);
  r.stop();
  CollectorSnapshot s = snapshot_from(st);
  CHECK(s.mem.total_mb > 0);
}

TEST(config_load_drops_and_clamps) {
  std::string base = make_tmp_dir();
  std::string presets_path = base + "/presets.json";
  write_file(presets_path,
             "[{\"name\":\"\",\"binary\":\"llama-bench\",\"args\":\"x\"},"
             "{\"name\":\"ok\",\"binary\":\"llama-bench\",\"args\":\"x\",\"repeats\":0},"
             "{\"name\":\"n\\u0000ul\",\"binary\":\"llama-bench\",\"args\":\"x\"}]");
  std::vector<Preset> presets;
  CHECK(load_presets(presets_path, presets));
  CHECK(presets.size() == 1);
  CHECK(presets[0].name == "ok" && presets[0].repeats == 1);
  std::string agents_path = base + "/agents.json";
  write_file(agents_path,
             "[{\"name\":\"a\",\"command\":\"\"},"
             "{\"name\":\"b\",\"command\":\"sh\",\"workdir\":\"/\"}]");
  std::vector<AgentSpec> agents;
  CHECK(load_agents(agents_path, agents));
  CHECK(agents.size() == 1 && agents[0].name == "b");
}

TEST(scan_llama_processes_fake_proc) {
  std::string base = make_tmp_dir();
  ensure_dir(base + "/1234");
  ensure_dir(base + "/1235");
  write_file(base + "/1234/comm", "llama-bench\n");
  write_file(base + "/1235/comm", "bash\n");
  auto pids = scan_llama_processes(base);
  CHECK(pids.size() == 1 && pids[0] == 1234);
}

TEST(parse_bench_json_fixture) {
  const char *bench =
      "[\n  {\"n_prompt\": 128, \"n_gen\": 0, \"avg_ts\": 157.667},\n"
      "  {\"n_prompt\": 0, \"n_gen\": 32, \"avg_ts\": 44.47}\n]\n";
  double tg = -2, pp = -2;
  CHECK(parse_llama_bench_json(bench, tg, pp));
  CHECK(pp > 157.6 && pp < 157.7);
  CHECK(tg > 44.4 && tg < 44.5);
  CHECK(!parse_llama_bench_json("not json", tg, pp));
}

TEST(run_manager_queue_edit) {
  std::string base = make_tmp_dir();
  RunManager rm(base, "/bin");
  Preset a;
  a.name = "a";
  a.binary = "/bin/echo";
  a.args = "x";
  a.repeats = 1;
  Preset b = a;
  b.name = "b";
  rm.enqueue({a});
  rm.enqueue({b});
  CHECK(rm.queue_snapshot().size() == 2);
  CHECK(rm.remove_queued(0));
  auto q = rm.queue_snapshot();
  CHECK(q.size() == 1 && q[0].preset.name == "b");
  CHECK(!rm.remove_queued(5));
}

TEST(run_manager_echo_record) {
  std::string base = make_tmp_dir();
  RunManager rm(base, "/bin");
  Preset p;
  p.name = "t 1";
  p.binary = "/bin/echo";
  p.args = "[{\"n_gen\":32,\"avg_ts\":44.5}]";
  p.repeats = 1;
  rm.enqueue({p});
  CHECK(rm.start());
  for (int i = 0; i < 100 && rm.records().empty(); ++i) usleep(50000);
  auto recs = rm.records();
  CHECK(recs.size() == 1);
  CHECK(recs[0].exit_code == 0);
  CHECK(recs[0].label == "t 1");
  CHECK(recs[0].tg_ts > 44.4 && recs[0].tg_ts < 44.6);
  CHECK(file_exists(recs[0].log_path) && file_exists(recs[0].output_json_path));
  bool ok = false;
  std::string line = slurp(base + "/runs.jsonl", ok);
  JsonValue v;
  CHECK(ok && json_parse(line, v));
  CHECK(v.find("label")->s == "t 1");
  // a finished run manager accepts a second start
  rm.enqueue({p});
  CHECK(rm.start());
  for (int i = 0; i < 100 && rm.records().size() < 2; ++i) usleep(50000);
  CHECK(rm.records().size() == 2);
}

TEST(run_manager_stop) {
  std::string base = make_tmp_dir();
  RunManager rm(base, "/bin");
  Preset p;
  p.name = "sleeper";
  p.binary = "/bin/sleep";
  p.args = "30";
  p.repeats = 1;
  rm.enqueue({p});
  CHECK(rm.start());
  for (int i = 0; i < 100 && !rm.status().active; ++i) usleep(20000);
  CHECK(rm.active_pid() > 0);
  rm.stop();
  for (int i = 0; i < 300 && rm.records().empty(); ++i) usleep(50000);
  auto recs = rm.records();
  CHECK(recs.size() == 1);
  CHECK(recs[0].exit_code == -15 || recs[0].exit_code == -9);
  CHECK(recs[0].note == "stopped");
}

TEST(run_manager_detach) {
  std::string base = make_tmp_dir();
  pid_t pid = -1;
  {
    RunManager rm(base, "/bin");
    Preset p;
    p.name = "sleeper";
    p.binary = "/bin/sleep";
    p.args = "30";
    p.repeats = 1;
    rm.enqueue({p});
    CHECK(rm.start());
    for (int i = 0; i < 100 && !rm.status().active; ++i) usleep(20000);
    pid = (pid_t)rm.active_pid();
    CHECK(pid > 0);
    rm.detach();
    for (int i = 0; i < 200 && rm.status().active; ++i) usleep(50000);
    CHECK(!rm.status().active);
    auto recs = rm.records();
    CHECK(recs.size() == 1 && recs[0].note == "detached" && recs[0].exit_code == -1);
  }
  kill(pid, SIGKILL);  // clean up the detached child
  int st = 0;
  waitpid(pid, &st, 0);
}
