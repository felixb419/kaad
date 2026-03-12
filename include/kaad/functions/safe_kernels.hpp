#pragma once

#include <cmath>   // for log, exp, copysign, floor, round, sqrt
#include <cstdlib> // for abs
#include <limits>  // for numeric_limits

/**
 * @namespace kaad::Kernels
 * @brief Contains elementary operation kernels and their corresponding
 * gradients.
 */
namespace kaad::Kernels {

template <typename T>
static constexpr T epsilon =
    static_cast<T>(1000) * std::numeric_limits<T>::epsilon();

template <typename T>
static constexpr T max_finite = std::numeric_limits<T>::max();

template <typename T>
static constexpr T min_finite = std::numeric_limits<T>::min();

template <typename T> static inline const T max_exp = std::log(max_finite<T>);

template <typename T> static inline const T min_exp = std::log(min_finite<T>);

/**
 * @brief Binary kernel for division.
 * @tparam T The scalar type.
 */
template <typename T> struct safe_Div {
    using value_type = T;

    /**
     * @brief C = A / B
     */
    constexpr static void Op(T lhs, T rhs, T &res) noexcept {
        res =
            lhs / ((std::abs(rhs) < epsilon<T>) ? std::copysign(epsilon<T>, rhs)
                                                : rhs);
    }

    /**
     * @brief Computes the gradient of a division.
     */
    constexpr static void Grad(T lhs, T &d_lhs, T rhs, T &d_rhs,
                               [[maybe_unused]] T res, T d_res) noexcept {
        T rhs_safe =
            (std::abs(rhs) < epsilon<T>) ? std::copysign(epsilon<T>, rhs) : rhs;

        T rhs_inv = T(1) / rhs_safe;
        d_lhs += d_res * rhs_inv;

        d_rhs -= d_res * lhs * rhs_inv * rhs_inv;
    }
};

/**
 * @brief Binary kernel for power.
 * @tparam T The scalar type.
 */
template <typename T> struct safe_Pow {
    using value_type = T;

    /**
     * @brief C = A ^ B
     */
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
        if (rhs_loglhs > max_exp<T>)
            res = max_finite<T>;
        else if (rhs_loglhs < min_exp<T>)
            res = 0;
        else {
            res = std::exp(rhs_loglhs);
            if (lhs < 0)
                res *= (int(std::floor(rhs + 0.5)) % 2 == 0 ? 1 : -1);
        }
    }
    /**
     * @brief Computes the gradient of an exponentiation.
     */
    constexpr static void Grad(T lhs, T &d_lhs, [[maybe_unused]] T rhs,
                               T &d_rhs, T res, T d_res) noexcept {
        if (lhs == 0 || std::abs(lhs) < epsilon<T>) {
            return;
        }
        d_lhs += d_res * (res / lhs);
        d_rhs += d_res * res * std::log(std::abs(lhs));
    }
};

/**
 * @brief Unary kernel for squareroot.
 * @tparam T The scalar type.
 */
template <typename T> struct safe_Sqrt {
    using value_type = T;

    /**
     * @brief C = sqrt(A)
     */
    constexpr static void Op(T lhs, T &res) noexcept {
        res = std::sqrt(std::max(lhs, T(0)));
    }
    constexpr static void Grad([[maybe_unused]] T lhs, T &d_lhs, T res,
                               T d_res) noexcept {
        /**
         * @brief Computes the gradient of the squareroot
         */
        d_lhs += res < epsilon<T> ? 0 : d_res / (2 * res);
    }
};

/**
 * @brief Unary kernel for logarithm.
 * @tparam T The scalar type.
 */
template <typename T> struct safe_Log {
    using value_type = T;

    /**
     * @brief C = log(A)
     */
    constexpr static void Op(T lhs, T &res) noexcept {
        res = std::log(std::max(lhs, epsilon<T>));
    }
    /**
     * @brief Computes the gradient of the logarithm
     */
    constexpr static void Grad(T lhs, T &d_lhs, [[maybe_unused]] T res,
                               T d_res) noexcept {
        d_lhs += d_res / std::max(lhs, epsilon<T>);
    }
};

/**
 * @brief Unary kernel for exponent.
 * @tparam T The scalar type.
 */
template <typename T> struct safe_Exp {
    using value_type = T;

    constexpr static void Op(T lhs, T &res) noexcept {
        /**
         * @brief C = e^A
         */
        res = std::exp(std::max(min_exp<T>, std::min(max_exp<T>, lhs)));
    }
    /**
     * @brief Computes the gradient of the exp function.
     */
    constexpr static void Grad([[maybe_unused]] T lhs, T &d_lhs, T res,
                               T d_res) noexcept {
        d_lhs += d_res * res;
    }
};

} // namespace kaad::Kernels
