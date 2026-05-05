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

/// Relative tolerance threshold: 1000× machine EPSILON.
template <typename T>
static constexpr T EPSILON =
    static_cast<T>(1000) * std::numeric_limits<T>::epsilon();

/// Largest finite representable value.
template <typename T>
static constexpr T MAX_FINITE = std::numeric_limits<T>::max();

/// Smallest positive finite representable value.
template <typename T>
static constexpr T MIN_FINITE = std::numeric_limits<T>::min();

// NOLINTBEGIN(bugprone-throwing-static-initialization)
// std::log will never throw here since MAX_FINITE and MIN_FINITE are always
// positive for floating point types.

/// Maximum safe exponent: log(MAX_FINITE).
template <typename T> static inline const T MAX_EXP = std::log(MAX_FINITE<T>);

/// Minimum safe exponent: log(MIN_FINITE).
template <typename T> static inline const T MIN_EXP = std::log(MIN_FINITE<T>);

// NOLINTEND(bugprone-throwing-static-initialization)

/// Numerically safe division kernel with forward/backward operations.
/// @ingroup binary_kernels
template <typename T> struct SafeDiv {
    using value_type = T;

    /// Forward op: res = lhs / max(rhs, EPSILON)
    constexpr static void op(T lhs, T rhs, T &res) noexcept {
        res =
            lhs / ((std::abs(rhs) < EPSILON<T>) ? std::copysign(EPSILON<T>, rhs)
                                                : rhs);
    }

    /// Backward op: accumulates d_res into d_lhs and d_rhs.
    constexpr static void grad(T lhs, T &d_lhs, T rhs, T &d_rhs,
                               [[maybe_unused]] T res, T d_res) noexcept {
        T rhs_safe =
            (std::abs(rhs) < EPSILON<T>) ? std::copysign(EPSILON<T>, rhs) : rhs;

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
    constexpr static void op(T lhs, T rhs, T &res) noexcept {
        if (lhs == 0) {
            res = (rhs == 0 ? 1 : 0); // 0^0=1 policy
            return;
        }

        if (lhs < 0 && std::abs(rhs - std::round(rhs)) > EPSILON<T>) {
            res = 0; // or MAX_FINITE * sign, but 0 is simple
            return;
        }

        T abs_lhs = std::abs(lhs);
        T rhs_loglhs = rhs * std::log(abs_lhs);

        if (rhs_loglhs > MAX_EXP<T>) {

            res = MAX_FINITE<T>;

        } else if (rhs_loglhs < MIN_EXP<T>) {

            res = 0;

        } else {
            res = std::exp(rhs_loglhs);
            if (lhs < 0) {

                res *= (std::lround(rhs) % 2 == 0 ? 1 : -1);
            }
        }
    }

    /// Backward op: accumulates d_res into d_lhs and d_rhs.
    constexpr static void grad(T lhs, T &d_lhs, [[maybe_unused]] T rhs,
                               T &d_rhs, T res, T d_res) noexcept {
        if (lhs == 0 || std::abs(lhs) < EPSILON<T>) {
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
    constexpr static void op(T inp, T &res) noexcept {
        res = std::sqrt(std::max(inp, T(0)));
    }
    /// Backward op: accumulates d_res into d_inp.
    constexpr static void grad([[maybe_unused]] T inp, T &d_inp, T res,
                               T d_res) noexcept {

        d_inp += res < EPSILON<T> ? 0 : d_res / (2 * res);
    }
};

/// Numerically safe logarithm kernel with forward/backward operations.
/// @ingroup unary_kernels
template <typename T> struct SafeLog {
    using value_type = T;

    /// Forward op: res = log(max(inp, EPSILON))
    constexpr static void op(T inp, T &res) noexcept {
        res = std::log(std::max(inp, EPSILON<T>));
    }

    /// Backward op: accumulates d_res into d_inp.
    constexpr static void grad(T inp, T &d_inp, [[maybe_unused]] T res,
                               T d_res) noexcept {
        d_inp += d_res / std::max(inp, EPSILON<T>);
    }
};

/// Numerically safe exponent kernel with forward/backward operations.
/// @ingroup unary_kernels
template <typename T> struct SafeExp {
    using value_type = T;

    /// Forward op: res = e ^ (max(MIN_EXP, min(MAX_EXP, inp)))
    constexpr static void op(T inp, T &res) noexcept {

        res = std::exp(std::max(MIN_EXP<T>, std::min(MAX_EXP<T>, inp)));
    }

    /// Backward op: accumulates d_res into d_inp.
    constexpr static void grad([[maybe_unused]] T inp, T &d_inp, T res,
                               T d_res) noexcept {
        d_inp += d_res * res;
    }
};

} // namespace kaad::Kernels
