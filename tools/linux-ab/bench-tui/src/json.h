#pragma once
// Minimal JSON value for bench-tui's flat config and record files.
// Supports null, bool, int64, double, string, array, object. Not a general
// JSON library; malformed or trailing content is a parse error.
// \uXXXX escapes above the BMP (surrogate pairs) are not combined (minimal-scope limitation).
#include <map>
#include <string>
#include <vector>

struct JsonValue {
  enum class Type { Null, Bool, Int, Double, Str, Arr, Obj };
  Type type = Type::Null;
  bool b = false;
  long long i = 0;
  double d = 0.0;
  std::string s;
  std::vector<JsonValue> arr;
  std::map<std::string, JsonValue> obj;

  static JsonValue make_null();
  static JsonValue make_bool(bool v);
  static JsonValue make_int(long long v);
  static JsonValue make_double(double v);
  static JsonValue make_str(std::string v);
  static JsonValue make_arr(std::vector<JsonValue> v);
  static JsonValue make_obj(std::map<std::string, JsonValue> v);
  const JsonValue *find(const std::string &key) const;  // nullptr when absent or not an object
  double as_number() const;  // Int -> i, Double -> d, anything else -> 0
};

bool json_parse(const std::string &text, JsonValue &out);  // false on malformed input
std::string json_stringify(const JsonValue &v);            // compact; object keys sorted
std::string json_escape(const std::string &in);
