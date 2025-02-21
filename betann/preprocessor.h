#ifndef BETANN_PREPROCESSOR_H_
#define BETANN_PREPROCESSOR_H_

#include <cstdint>
#include <map>
#include <string>
#include <variant>

namespace betann {

using VariablesMap = std::map<std::string_view,
                              std::variant<std::string_view, bool, uint32_t>>;
// Provide a template string |templ|, return a new string that does following
// replacements:
// * Words like "$name" are replaced by |variables|.
// * The content in "if ($cond) { ... }" are removed if $cond is false.
std::string ParseTemplate(std::string_view templ,
                          const VariablesMap& variables);

template<typename... Args>
inline std::string ParseTemplate(std::string_view templ,
                                 VariablesMap variables,
                                 Args... args) {
  (variables.merge(args), ...);
  return ParseTemplate(templ, variables);
}

}  // namespace betann

#endif  // BETANN_PREPROCESSOR_H_
