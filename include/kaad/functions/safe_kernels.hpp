#pragma once

#include <cmath>   // for log, exp, copysign, floor, round, sqrt, lround
#include <cstdlib> // for abs
#include <limits>  // for numeric_limits

/**
 * @namespace kaad::Kernels
 * @brief Contains elementary operation kernels and their corresponding
 * gradients.
 */
namespace kaad::Kernels {

/// Relative tolerance threshold: 1000× machine epsilon.
template <typename T>
static constexpr T epsilon =
    static_cast<T>(1000) * std::numeric_limits<T>::epsilon();

/// Largest finite representable value.
template <typename T>
static constexpr T max_finite = std::numeric_limits<T>::max();

/// Smallest positive finite representable value.
template <typename T>
static constexpr T min_finite = std::numeric_limits<T>::min();

// NOLINTBEGIN(bugprone-throwing-static-initialization)
// std::log will never throw here since max_finite and min_finite are always
// positive for floating point types.

/// Maximum safe exponent: log(max_finite).
template <typename T> static inline const T max_exp = std::log(max_finite<T>);

/// Minimum safe exponent: log(min_finite).
template <typename T> static inline const T min_exp = std::log(min_finite<T>);

// NOLINTEND(bugprone-throwing-static-initialization)

/// Numerically safe division kernel with forward/backward operations.
/// @ingroup binary_kernels
template <typename T> struct SafeDiv {
    using value_type = T;

    /// Forward op: res = lhs / max(rhs, epsilon)
    constexpr static void Op(T lhs, T rhs, T &res) noexcept {
        res =
            lhs / ((std::abs(rhs) < epsilon<T>) ? std::copysign(epsilon<T>, rhs)
                                                : rhs);
    }

    /// Backward op: accumulates d_res into d_lhs and d_rhs.
    constexpr static void Grad(T lhs, T &d_lhs, T rhs, T &d_rhs,
                               [[maybe_unused]] T res, T d_res) noexcept {
        T rhs_safe =
            (std::abs(rhs) < epsilon<T>) ? std::copysign(epsilon<T>, rhs) : rhs;

        T rhs_inv = T(1) / rhs_safe;
        d_lhs += d_res * rhs_inv;

        d_rhs -= d_res * lhs * rhs_inv * rhs_inv;
    }
};

/// Numerically safe power kernel with forward/backward operations.
/// @ingroup binary_kernels
template <typename T> struct SafePow {
    using value_type = T;

    /// Forward op: res = e ^ (rhs * log(abs(lhs)))
    constexpr static void Op(T lhs, T rhs, T &res) noexcept {
        if (lhs == 0) {
            res = (rhs == 0 ? 1 : 0); // 0^0=1 policy
            return;
        }

        if (lhs < 0 && std::abs(rhs - std::round(rhs)) > epsilon<T>) {
            res = 0; // or max_finite * sign, but 0 is simple
            return;
        }

        T abs_lhs = std::abs(lhs);
        T rhs_loglhs = rhs * std::log(abs_lhs);

        if (rhs_loglhs > max_exp<T>) {

            res = max_finite<T>;

        } else if (rhs_loglhs < min_exp<T>) {

            res = 0;

        } else {
            res = std::exp(rhs_loglhs);
            if (lhs < 0) {

                res *= (std::lround(rhs) % 2 == 0 ? 1 : -1);
            }
        }
    }

    /// Backward op: accumulates d_res into d_lhs and d_rhs.
    constexpr static void Grad(T lhs, T &d_lhs, [[maybe_unused]] T rhs,
                               T &d_rhs, T res, T d_res) noexcept {
        if (lhs == 0 || std::abs(lhs) < epsilon<T>) {
            return;
        }
        d_lhs += d_res * (res / lhs);
        d_rhs += d_res * res * std::log(std::abs(lhs));
    }
};

/// Numerically safe squareroot kernel with forward/backward operations.
/// @ingroup unary_kernels
template <typename T> struct SafeSqrt {
    using value_type = T;

    /// Forward op: res = sqrt(max(inp, 0))
    constexpr static void Op(T inp, T &res) noexcept {
        res = std::sqrt(std::max(inp, T(0)));
    }
    /// Backward op: accumulates d_res into d_inp.
    constexpr static void Grad([[maybe_unused]] T inp, T &d_inp, T res,
                               T d_res) noexcept {

        d_inp += res < epsilon<T> ? 0 : d_res / (2 * res);
    }
};

/// Numerically safe logarithm kernel with forward/backward operations.
/// @ingroup unary_kernels
template <typename T> struct SafeLog {
    using value_type = T;

    /// Forward op: res = log(max(inp, epsilon))
    constexpr static void Op(T inp, T &res) noexcept {
        res = std::log(std::max(inp, epsilon<T>));
    }

    /// Backward op: accumulates d_res into d_inp.
    constexpr static void Grad(T inp, T &d_inp, [[maybe_unused]] T res,
                               T d_res) noexcept {
        d_inp += d_res / std::max(inp, epsilon<T>);
    }
};

/// Numerically safe exponent kernel with forward/backward operations.
/// @ingroup unary_kernels
template <typename T> struct SafeExp {
    using value_type = T;

    /// Forward op: res = e ^ (max(min_exp, min(max_exp, inp)))
    constexpr static void Op(T inp, T &res) noexcept {

        res = std::exp(std::max(min_exp<T>, std::min(max_exp<T>, inp)));
    }

    /// Backward op: accumulates d_res into d_inp.
    constexpr static void Grad([[maybe_unused]] T inp, T &d_inp, T res,
                               T d_res) noexcept {
        d_inp += d_res * res;
    }
};

} // namespace kaad::Kernels
