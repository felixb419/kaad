#pragma once

#include <initializer_list> // for initializer_list
#include <span>             // for span
#include <stdexcept>        // for runtime_error
#include <string>           // for string
#include <utility>          // for pair

namespace kaad {

class kaad_error : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

class argument_error : public kaad_error {
  public:
    inline static const char *err_type = "argument error";

    using kaad_error::kaad_error;
};

class shape_error : public kaad_error {
  public:
    inline static const char *err_type = "shape error";

    using kaad_error::kaad_error;
};

std::string make_graph_errmsg(
    const char *err_type, std::size_t graph_idx, const char *op_name,
    const char *msg,
    std::initializer_list<std::pair<const char *, std::span<const int>>>
        arrays = std::initializer_list<
            std::pair<const char *, std::span<const int>>>{},
    std::initializer_list<std::pair<const char *, int>> numbers =
        std::initializer_list<std::pair<const char *, int>>{});

} // namespace kaad
