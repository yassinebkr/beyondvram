// bench-tui unit tests. Single-file CHECK runner; TEST blocks self-register.
// Exit code is 0 when nothing failed, 1 otherwise.
#include <cstdio>
#include <deque>
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
