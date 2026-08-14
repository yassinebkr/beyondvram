// bench-tui unit tests. Single-file CHECK runner; TEST blocks self-register.
// Exit code is 0 when nothing failed, 1 otherwise.
#include "json.h"
#include <cstdio>
#include <deque>
#include <map>
#include <string>
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
