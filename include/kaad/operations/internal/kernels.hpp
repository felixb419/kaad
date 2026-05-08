#pragma once

#include <cmath>   // for log, pow, exp, sqrt
#include <cstdlib> // for abs

/**
 * @namespace kaad::operations::kernels
 * @brief Contains elementary operation kernels and their corresponding
 * gradients.
 */
namespace kaad::operations::kernels {

/// @defgroup kernels Kernels for elementary operations and their corresponding
/// gradients.

/// @defgroup binary_kernels Kernels that take two inputs.
/// @ingroup kernels

/// @defgroup unary_kernels
/// @ingroup kernels

/**
 * @brief Concept requiring a kernel to have:
 * 1. 'value_type' alias
 * 2. static void Op(const value_type&, const value_type&, value_type&);
 * 3. static void Grad(const value_type&, value_type&, const value_type&,
 *                   value_type&, const value_type&, const value_type&);
 * @ingroup binary_kernels
 */
template <class Kernel>
concept Binary = requires { typename Kernel::value_type; } && requires {
    {
        Kernel::op(std::declval<const typename Kernel::value_type &>(),
                   std::declval<const typename Kernel::value_type &>(),
                   std::declval<typename Kernel::value_type &>())
    } -> std::same_as<void>;
    {
        Kernel::grad(std::declval<const typename Kernel::value_type &>(),
                     std::declval<typename Kernel::value_type &>(),
                     std::declval<const typename Kernel::value_type &>(),
                     std::declval<typename Kernel::value_type &>(),
                     std::declval<const typename Kernel::value_type &>(),
                     std::declval<const typename Kernel::value_type &>())
    } -> std::same_as<void>;
};

template <Binary Kernel> constexpr bool binary_noexcept() {
    return noexcept(
               Kernel::op(std::declval<const typename Kernel::value_type &>(),
                          std::declval<const typename Kernel::value_type &>(),
                          std::declval<typename Kernel::value_type &>())) &&
           noexcept(Kernel::grad(
               std::declval<const typename Kernel::value_type &>(),
               std::declval<typename Kernel::value_type &>(),
               std::declval<const typename Kernel::value_type &>(),
               std::declval<typename Kernel::value_type &>(),
               std::declval<const typename Kernel::value_type &>(),
               std::declval<const typename Kernel::value_type &>()));
}

/**
 * @brief Concept requiring a kernel to have:
 * 1. 'value_type' alias
 * 2. static void Op(const value_type&, value_type&);
 * 3. static void Grad(const value_type&, value_type&, const value_type&, const
 * value_type&);
 */
template <class Kernel>
concept Unary = requires { typename Kernel::value_type; } && requires {
    {
        Kernel::op(std::declval<const typename Kernel::value_type &>(),
                   std::declval<typename Kernel::value_type &>())
    } -> std::same_as<void>;
    {
        Kernel::grad(std::declval<const typename Kernel::value_type &>(),
                     std::declval<typename Kernel::value_type &>(),
                     std::declval<const typename Kernel::value_type &>(),
                     std::declval<const typename Kernel::value_type &>())
    } -> std::same_as<void>;
};

template <Unary Kernel> constexpr bool unary_noexcept() {
    return noexcept(Kernel::grad(
               std::declval<const typename Kernel::value_type &>(),
               std::declval<typename Kernel::value_type &>(),
               std::declval<const typename Kernel::value_type &>(),
               std::declval<const typename Kernel::value_type &>())) &&
           noexcept(
               Kernel::op(std::declval<const typename Kernel::value_type &>(),
                          std::declval<typename Kernel::value_type &>()));
}

/// Elementwise addition kernel with forward and backward ops.
/// @ingroup binary_kernels
template <typename T> struct Add {
    using value_type = T;

    /// Forward op: res = lhs + rhs.
    constexpr static void op(T lhs, T rhs, T &res) noexcept { res = lhs + rhs; }

    /// Backward op: accumulates d_res into d_lhs and d_rhs.
    constexpr static void grad([[maybe_unused]] T lhs, T &d_lhs,
                               [[maybe_unused]] T rhs, T &d_rhs,
                               [[maybe_unused]] T res, T d_res) noexcept {
        d_lhs += d_res;
        d_rhs += d_res;
    }
};

/// Elementwise subtraction kernel with forward and backward ops.
/// @ingroup binary_kernels
template <typename T> struct Sub {
    using value_type = T;

    /// Forward op: res = lhs - rhs.
    constexpr static void op(T lhs, T rhs, T &res) noexcept { res = lhs - rhs; }

    /// Backward op: accumulates d_res into d_lhs and d_rhs.
    constexpr static void grad([[maybe_unused]] T lhs, T &d_lhs,
                               [[maybe_unused]] T rhs, T &d_rhs,
                               [[maybe_unused]] T res, T d_res) noexcept {
        d_lhs += d_res;
        d_rhs -= d_res;
    }
};

/// Elementwise multiplication kernel with forward and backward ops.
/// @ingroup binary_kernels
template <typename T> struct Mul {
    using value_type = T;

    /// Forward op: res = lhs * rhs.
    constexpr static void op(T lhs, T rhs, T &res) noexcept { res = lhs * rhs; }

    /// Backward op: accumulates d_res into d_lhs and d_rhs.
    constexpr static void grad(T lhs, T &d_lhs, T rhs, T &d_rhs,
                               [[maybe_unused]] T res, T d_res) noexcept {
        d_lhs += d_res * rhs;
        d_rhs += d_res * lhs;
    }
};

/// Elementwise division kernel with forward and backward ops.
/// @ingroup binary_kernels
template <typename T> struct Div {
    using value_type = T;

    /// Forward op: res = lhs / rhs.
    constexpr static void op(T lhs, T rhs, T &res) noexcept { res = lhs / rhs; }

    /// Backward op: accumulates d_res into d_lhs and d_rhs.
    constexpr static void grad(T lhs, T &d_lhs, T rhs, T &d_rhs,
                               [[maybe_unused]] T res, T d_res) noexcept {
        d_lhs += d_res * (1 / rhs);
        d_rhs -= d_res * (lhs / (rhs * rhs));
    }
};

/// Elementwise power kernel with forward and backward ops.
/// @ingroup binary_kernels
template <typename T> struct Pow {
    using value_type = T;

    /// Forward op: res = lhs ^ rhs.
    constexpr static void op(T lhs, T rhs, T &res) noexcept {
        res = std::pow(lhs, rhs);
    }

    /// Backward op: accumulates d_res into d_lhs and d_rhs.
    constexpr static void grad(T lhs, T &d_lhs, T rhs, T &d_rhs, T res,
                               T d_res) noexcept {
        d_lhs += d_res * rhs * std::pow(lhs, rhs - 1);
        d_rhs += d_res * res * std::log(lhs);
    }
};

/// Elementwise dot product kernel with forward and backward ops.
/// @ingroup binary_kernels
template <typename T> struct Dot {
    using value_type = T;

    /// Forward op: res += lhs + rhs.
    constexpr static void op(T lhs, T rhs, T &res) noexcept {
        res += lhs * rhs;
    }

    /// Backward op: accumulates d_res into d_lhs and d_rhs.
    constexpr static void grad(T lhs, T &d_lhs, T rhs, T &d_rhs,
                               [[maybe_unused]] T res, T d_res) noexcept {
        d_lhs += d_res * rhs;
        d_rhs += d_res * lhs;
    }
};

/// Elementwise minimum kernel with forward and backward ops.
/// @ingroup binary_kernels
template <typename T> struct Min {
    using value_type = T;

    /// Forward op: res = min(lhs, rhs)
    constexpr static void op(T lhs, T rhs, T &res) noexcept {
        res = lhs < rhs ? lhs : rhs;
    }

    /// Backward op: accumulates d_res into d_lhs and d_rhs.
    constexpr static void grad(T lhs, T &d_lhs, T rhs, T &d_rhs,
                               [[maybe_unused]] T res, T d_res) noexcept {
        bool smaller = lhs <= rhs;
        d_lhs += smaller ? d_res : 0;
        d_rhs += smaller ? 0 : d_res;
    }
};

/// Elementwise maximum kernel with forward and backward ops.
/// @ingroup binary_kernels
template <typename T> struct Max {
    using value_type = T;

    /// Forward op: res = max(lhs, rhs)
    constexpr static void op(T lhs, T rhs, T &res) noexcept {
        res = lhs > rhs ? lhs : rhs;
    }

    /// Backward op: accumulates d_res into d_lhs and d_rhs.
    constexpr static void grad(T lhs, T &d_lhs, T rhs, T &d_rhs,
                               [[maybe_unused]] T res, T d_res) noexcept {
        bool bigger = lhs >= rhs;
        d_lhs += bigger ? d_res : 0;
        d_rhs += bigger ? 0 : d_res;
    }
};

/// Elementwise kernel for 'no operation' with forward and backward ops.
/// @ingroup unary_kernels
template <typename T> struct NoOp {
    using value_type = T;

    /// Forward op: res = inp
    constexpr static void op(T inp, T &res) noexcept { res = inp; }

    /// Backward op: accumulates d_res into d_inp.
    constexpr static void grad([[maybe_unused]] T inp, T &d_inp,
                               [[maybe_unused]] T res, T d_res) noexcept {
        d_inp += d_res;
    }
};

/// Elementwise negation kernel with forward and backward ops.
/// @ingroup unary_kernels
template <typename T> struct Neg {
    using value_type = T;

    /// Forward op: res = -inp
    constexpr static void op(T inp, T &res) noexcept { res = -inp; }

    /// Backward op: accumulates d_res into d_inp.
    constexpr static void grad([[maybe_unused]] T inp, T &d_inp,
                               [[maybe_unused]] T res, T d_res) noexcept {
        d_inp -= d_res;
    }
};

/// Elementwise square kernel with forward and backward ops.
/// @ingroup unary_kernels
template <typename T> struct Square {
    using value_type = T;

    /// Forward op: res = inp ^ 2
    constexpr static void op(T inp, T &res) noexcept { res = inp * inp; }

    /// Backward op: accumulates d_res into d_inp.
    constexpr static void grad(T inp, T &d_inp, [[maybe_unused]] T res,
                               T d_res) noexcept {
        d_inp += d_res * 2 * inp;
    }
};

/// Elementwise squareroot kernel with forward and backward ops.
/// @ingroup unary_kernels
template <typename T> struct Sqrt {
    using value_type = T;

    /// Forward op: res = sqrt(inp)
    constexpr static void op(T inp, T &res) noexcept { res = std::sqrt(inp); }

    /// Backward op: accumulates d_res into d_inp.
    constexpr static void grad([[maybe_unused]] T inp, T &d_inp, T res,
                               T d_res) noexcept {
        d_inp += d_res / (2 * res);
    }
};

/// Elementwise logarithm kernel with forward and backward ops.
/// @ingroup unary_kernels
template <typename T> struct Log {
    using value_type = T;

    /// Forward op: res = log(inp)
    constexpr static void op(T inp, T &res) noexcept { res = std::log(inp); }

    /// Backward op: accumulates d_res into d_inp.
    constexpr static void grad(T inp, T &d_inp, [[maybe_unused]] T res,
                               T d_res) noexcept {
        d_inp += d_res / inp;
    }
};

/// Elementwise exponent kernel with forward and backward ops.
/// @ingroup unary_kernels
template <typename T> struct Exp {
    using value_type = T;

    /// Forward op: res = e ^ inp
    constexpr static void op(T inp, T &res) noexcept { res = std::exp(inp); }

    /// Backward op: accumulates d_res into d_inp.
    constexpr static void grad([[maybe_unused]] T inp, T &d_inp, T res,
                               T d_res) noexcept {
        d_inp += d_res * res;
    }
};

/// Elementwise absolute value kernel with forward and backward ops.
/// @ingroup unary_kernels
template <typename T> struct Abs {
    using value_type = T;

    /// Forward op: res = | inp |
    constexpr static void op(T inp, T &res) noexcept { res = std::abs(inp); }

    /// Backward op: accumulates d_res into d_inp.
    constexpr static void grad(T inp, T &d_inp, [[maybe_unused]] T res,
                               T d_res) noexcept {
        d_inp += d_res * (inp > 0 ? 1 : -1);
    }
};

} // namespace kaad::operations::kernels
