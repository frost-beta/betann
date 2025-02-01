#include "betann/preprocessor.h"

#include <fmt/format.h>

namespace betann {

namespace {

size_t FindEndOfVar(std::string_view templ, size_t pos) {
  return templ.find_first_not_of("abcdefghijklmnopqrstuvwxyz_", pos);
}

VariablesMap::mapped_type GetVar(const VariablesMap& variables,
                                 std::string_view templ,
                                 size_t pos, size_t end) {
  std::string_view varName = templ.substr(pos, end - pos);
  auto var = variables.find(varName);
  if (var == variables.end())
    throw std::runtime_error(fmt::format("Variable not found: {}", varName));
  return var->second;
}

std::tuple<size_t, size_t, std::string_view> GetBraceBody(
    std::string_view templ, size_t pos) {
  size_t start = templ.find("{", pos);
  if (start == std::string::npos)
    throw std::runtime_error("Can not find { after if statement.");
  size_t end = templ.find("}", start);
  if (end == std::string::npos)
    throw std::runtime_error("Can not find matching } for if statement.");
  start += 1;
  return {start, end, templ.substr(start, end - start)};
}

std::string KeepOnlyNewLines(std::string_view text) {
  return std::string(std::count(text.begin(), text.end(), '\n'), '\n');
}

}  // namespace

std::string ParseTemplate(std::string_view templ,
                          const VariablesMap& variables) {
  std::string result;
  // First handle all "if" statements.
  while (templ.size() > 0) {
    size_t start = templ.find("if (");
    if (start >= templ.size() - 6) {
      result += templ;
      break;
    }
    size_t pos = start + 4;
    if (templ[pos] != '$' && templ.substr(pos, 2) != "!$") {
      result += templ.substr(0, pos);
      templ = templ.substr(pos);
      continue;
    }
    result += templ.substr(0, start);
    bool no = templ[pos] == '!';
    pos += 1 + no;
    auto var = GetVar(variables, templ, pos, FindEndOfVar(templ, pos));
    if (!std::holds_alternative<bool>(var))
      throw std::runtime_error("Variable in condition must be bool.");
    auto [content, end, body] = GetBraceBody(templ, pos);
    bool condition = std::get<bool>(var) != no;
    result += KeepOnlyNewLines(templ.substr(start, content - start));
    result += condition ? body : KeepOnlyNewLines(body);
    if (templ.substr(end, 8) == "} else {") {
      auto [_, elseEnd, elseBody] = GetBraceBody(templ, end + 7);
      result += std::string(8, ' ');
      result += condition ? KeepOnlyNewLines(elseBody) : elseBody;
      templ = templ.substr(elseEnd + 1);
    } else {
      templ = templ.substr(end + 1);
    }
  }
  // Then handle all variable replacements.
  std::string result2;
  templ = result;
  while (templ.size() > 0) {
    size_t pos = templ.find('$');
    if (pos >= templ.size() - 1) {
      result2 += templ;
      break;
    }
    result2 += templ.substr(0, pos);
    pos += 1;
    size_t end = FindEndOfVar(templ, pos);
    auto var = GetVar(variables, templ, pos, end);
    std::visit([&result2](auto&& arg) {
      result2 += fmt::format("{}", arg);
    }, var);
    templ = templ.substr(end);
  }
  return result2;
}

}  // namespace betann
