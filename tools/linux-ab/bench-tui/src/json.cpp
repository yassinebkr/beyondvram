#include "json.h"
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

JsonValue JsonValue::make_null() { return JsonValue(); }
JsonValue JsonValue::make_bool(bool v) { JsonValue x; x.type = Type::Bool; x.b = v; return x; }
JsonValue JsonValue::make_int(long long v) { JsonValue x; x.type = Type::Int; x.i = v; return x; }
JsonValue JsonValue::make_double(double v) { JsonValue x; x.type = Type::Double; x.d = v; return x; }
JsonValue JsonValue::make_str(std::string v) { JsonValue x; x.type = Type::Str; x.s = std::move(v); return x; }
JsonValue JsonValue::make_arr(std::vector<JsonValue> v) { JsonValue x; x.type = Type::Arr; x.arr = std::move(v); return x; }
JsonValue JsonValue::make_obj(std::map<std::string, JsonValue> v) { JsonValue x; x.type = Type::Obj; x.obj = std::move(v); return x; }

const JsonValue *JsonValue::find(const std::string &key) const {
  if (type != Type::Obj) return nullptr;
  auto it = obj.find(key);
  return it == obj.end() ? nullptr : &it->second;
}

double JsonValue::as_number() const {
  if (type == Type::Int) return (double)i;
  if (type == Type::Double) return d;
  return 0.0;
}

std::string json_escape(const std::string &in) {
  std::string out;
  for (char c : in) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if ((unsigned char)c < 0x20) {
          char b[8];
          std::snprintf(b, sizeof b, "\\u%04x", c);
          out += b;
        } else {
          out += c;
        }
    }
  }
  return out;
}

std::string json_stringify(const JsonValue &v) {
  switch (v.type) {
    case JsonValue::Type::Null: return "null";
    case JsonValue::Type::Bool: return v.b ? "true" : "false";
    case JsonValue::Type::Int: return std::to_string(v.i);
    case JsonValue::Type::Double: {
      char b[32];
      std::snprintf(b, sizeof b, "%.17g", v.d);
      return b;
    }
    case JsonValue::Type::Str: return "\"" + json_escape(v.s) + "\"";
    case JsonValue::Type::Arr: {
      std::string r = "[";
      bool first = true;
      for (const auto &e : v.arr) {
        if (!first) r += ",";
        first = false;
        r += json_stringify(e);
      }
      return r + "]";
    }
    case JsonValue::Type::Obj: {
      std::string r = "{";
      bool first = true;
      for (const auto &kv : v.obj) {
        if (!first) r += ",";
        first = false;
        r += "\"" + json_escape(kv.first) + "\":" + json_stringify(kv.second);
      }
      return r + "}";
    }
  }
  return "null";
}

namespace {
struct Parser {
  const char *p;
  int depth = 0;
  // Nesting deeper than 128 levels is rejected to bound parser stack use.
  struct Enter {
    Parser &x;
    bool ok;
    explicit Enter(Parser &x) : x(x), ok(x.depth < 128) { if (ok) ++x.depth; }
    ~Enter() { if (ok) --x.depth; }
  };
  void ws() { while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') ++p; }
  bool lit(const char *w) {
    size_t n = std::strlen(w);
    if (std::strncmp(p, w, n) == 0) { p += n; return true; }
    return false;
  }
  bool string(std::string &out) {
    if (*p != '"') return false;
    ++p;
    std::string r;
    while (*p && *p != '"') {
      char c = *p++;
      if (c != '\\') { r += c; continue; }
      char e = *p++;
      if (!e) return false;
      switch (e) {
        case '"': r += '"'; break;
        case '\\': r += '\\'; break;
        case '/': r += '/'; break;
        case 'b': r += '\b'; break;
        case 'f': r += '\f'; break;
        case 'n': r += '\n'; break;
        case 'r': r += '\r'; break;
        case 't': r += '\t'; break;
        case 'u': {
          if (std::strlen(p) < 4) return false;
          unsigned cp = 0;
          for (int k = 0; k < 4; ++k) {
            char h = *p++;
            cp <<= 4;
            if (h >= '0' && h <= '9') cp |= (unsigned)(h - '0');
            else if (h >= 'a' && h <= 'f') cp |= (unsigned)(h - 'a' + 10);
            else if (h >= 'A' && h <= 'F') cp |= (unsigned)(h - 'A' + 10);
            else return false;
          }
          if (cp < 0x80) {
            r += (char)cp;
          } else if (cp < 0x800) {
            r += (char)(0xC0 | (cp >> 6));
            r += (char)(0x80 | (cp & 63));
          } else {
            r += (char)(0xE0 | (cp >> 12));
            r += (char)(0x80 | ((cp >> 6) & 63));
            r += (char)(0x80 | (cp & 63));
          }
          break;
        }
        default: return false;
      }
    }
    if (*p != '"') return false;
    ++p;
    out = std::move(r);
    return true;
  }
  bool number(JsonValue &out) {
    // Strict JSON grammar: -?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?
    const char *start = p;
    if (*p == '-') ++p;
    if (*p == '0') {
      ++p;
    } else if (*p >= '1' && *p <= '9') {
      while (*p >= '0' && *p <= '9') ++p;
    } else {
      return false;
    }
    bool is_int = true;
    if (*p == '.') {
      is_int = false;
      ++p;
      if (*p < '0' || *p > '9') return false;
      while (*p >= '0' && *p <= '9') ++p;
    }
    if (*p == 'e' || *p == 'E') {
      is_int = false;
      ++p;
      if (*p == '+' || *p == '-') ++p;
      if (*p < '0' || *p > '9') return false;
      while (*p >= '0' && *p <= '9') ++p;
    }
    if (is_int) {
      errno = 0;
      char *end = nullptr;
      long long iv = std::strtoll(start, &end, 10);
      if (errno == ERANGE || end != p) return false;
      out = JsonValue::make_int(iv);
    } else {
      char *end = nullptr;
      double dv = std::strtod(start, &end);
      if (end != p) return false;
      out = JsonValue::make_double(dv);
    }
    return true;
  }
  bool value(JsonValue &out) {
    ws();
    Enter enter(*this);
    if (!enter.ok) return false;
    if (*p == '"') {
      std::string s;
      if (!string(s)) return false;
      out = JsonValue::make_str(std::move(s));
      return true;
    }
    if (*p == '{') {
      ++p;
      std::map<std::string, JsonValue> m;
      ws();
      if (*p == '}') { ++p; out = JsonValue::make_obj(std::move(m)); return true; }
      for (;;) {
        ws();
        std::string k;
        if (!string(k)) return false;
        ws();
        if (*p != ':') return false;
        ++p;
        JsonValue v;
        if (!value(v)) return false;
        m[std::move(k)] = std::move(v);
        ws();
        if (*p == ',') { ++p; continue; }
        if (*p == '}') { ++p; break; }
        return false;
      }
      out = JsonValue::make_obj(std::move(m));
      return true;
    }
    if (*p == '[') {
      ++p;
      std::vector<JsonValue> a;
      ws();
      if (*p == ']') { ++p; out = JsonValue::make_arr(std::move(a)); return true; }
      for (;;) {
        JsonValue v;
        if (!value(v)) return false;
        a.push_back(std::move(v));
        ws();
        if (*p == ',') { ++p; continue; }
        if (*p == ']') { ++p; break; }
        return false;
      }
      out = JsonValue::make_arr(std::move(a));
      return true;
    }
    if (lit("true")) { out = JsonValue::make_bool(true); return true; }
    if (lit("false")) { out = JsonValue::make_bool(false); return true; }
    if (lit("null")) { out = JsonValue::make_null(); return true; }
    return number(out);
  }
};
}  // namespace

bool json_parse(const std::string &text, JsonValue &out) {
  Parser x{text.c_str()};
  if (!x.value(out)) return false;
  x.ws();
  return *x.p == '\0';
}
