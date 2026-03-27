#pragma once

#include <cstddef>          // for size_t
#include <initializer_list> // for initializer_list
#include <stdexcept>        // for runtime_error
#include <string>           // for string

// here to statisfy (likely buggy) iwyu
#include <span>    // for span // IWYU pragma: keep
#include <utility> // for pair // IWYU pragma: keep

namespace kaad {

std::string to_string(std::span<const int> array);

class KaadError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

class ArgumentError : public KaadError {
  public:
    inline static const char *err_type = "argument error";

    using KaadError::KaadError;
};

class ShapeError : public KaadError {
  public:
    inline static const char *err_type = "shape error";

    using KaadError::KaadError;
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
