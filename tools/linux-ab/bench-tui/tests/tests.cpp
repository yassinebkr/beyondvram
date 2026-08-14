// bench-tui unit tests. Single-file CHECK runner; TEST blocks self-register.
// Exit code is 0 when nothing failed, 1 otherwise.
#include "config.h"
#include "fsutil.h"
#include "json.h"
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <map>
#include <string>
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
