#pragma once

#include <cstddef>     // for size_t
#include <ranges>      // for range
#include <stdexcept>   // for runtime_error
#include <string>      // for basic_string, string, to_string
#include <string_view> // for string_view

namespace kaad {

/// @brief Base exception type for kaad.
struct Exception : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

/// @brief Exception for errors caused by conditions detected only at runtime.
struct RuntimeError : public Exception {
  public:
    using Exception::Exception;
};

/// @brief Exception for errors caused by incorrect API usage.
struct LogicError : public Exception {
  public:
    using Exception::Exception;
};

/// @brief Exception for errors caused by exceeding the capacity of a container.
struct CapacityError : public LogicError {
  public:
    using LogicError::LogicError;
};

/// @brief Exception for invalid or malformed arguments.
struct ArgumentError : public LogicError {
  public:
    inline static const char *err_type = "argument error";

    using LogicError::LogicError;
};

/// @brief Exception for invalid shapes.
struct ShapeError : public ArgumentError {
  public:
    inline static const char *err_type = "shape error";

    using ArgumentError::ArgumentError;
};

/// @brief Exception for errors that occur during shape broadcasting.
struct BroadcastError : public ShapeError {
  public:
    inline static const char *err_type = "broadcast error";

    using ShapeError::ShapeError;
};

/// @return A string containing the elements of @p array in square brackets.
template <std::ranges::input_range IR>
    requires requires { std::to_string(std::ranges::range_value_t<IR>{}); }
std::string to_string(const IR &range) {
    std::string str = "[";
    bool first = true;

    for (auto &&elem : range) {
        if (!first) {
            str += ", ";
        }
        first = false;
        str += std::to_string(elem);
    }

    str += "]";
    return str;
}

/// @return An error message formatted for a @ref Graph.
std::string make_graph_errmsg(std::size_t graph_idx, const char *op_name,
                              std::string_view msg);

} // namespace kaad
