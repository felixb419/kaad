#pragma once

#include <cstddef>          // for size_t
#include <initializer_list> // for initializer_list
#include <span>             // for span
#include <stdexcept>        // for runtime_error
#include <string>           // for string
#include <utility>          // for pair

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
std::string to_string(std::span<const int> array);

/// @return An error message formatted for a @ref Graph.
std::string make_graph_errmsg(
    std::size_t graph_idx, const char *op_name, const char *msg,
    std::initializer_list<std::pair<const char *, std::span<const int>>>
        arrays = std::initializer_list<
            std::pair<const char *, std::span<const int>>>{},
    std::initializer_list<std::pair<const char *, int>> numbers =
        std::initializer_list<std::pair<const char *, int>>{});

} // namespace kaad
