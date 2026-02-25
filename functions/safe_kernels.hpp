#pragma once

#include <cmath>   // for log, exp, copysign, floor, round, sqrt
#include <cstdlib> // for abs
#include <limits>  // for numeric_limits

namespace kaad {

/**
 * @namespace kaad::Kernels
 * @brief Contains elementary operation kernels and their corresponding
 * gradients.
 */
namespace Kernels {

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
    constexpr static void Op(T A, T B, T &C) noexcept {
        C = A / ((std::abs(B) < epsilon<T>) ? std::copysign(epsilon<T>, B) : B);
    }

    /**
     * @brief Computes the gradient of a division.
     */
    constexpr static void Grad(T A, T &dA, T B, T &dB, T C, T dC) noexcept {
        T B_safe =
            (std::abs(B) < epsilon<T>) ? std::copysign(epsilon<T>, B) : B;

        T B_inv = T(1) / B_safe;
        dA += dC * B_inv;

        dB -= dC * A * B_inv * B_inv;
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
    constexpr static void Op(T A, T B, T &C) noexcept {
        if (A == 0) {
            C = (B == 0 ? 1 : 0); // 0^0=1 policy
            return;
        }
        if (A < 0 && std::abs(B - std::round(B)) > epsilon<T>) {
            C = 0; // or max_finite * sign, but 0 is simple
            return;
        }
        T absA = std::abs(A);
        T t = B * std::log(absA);
        if (t > max_exp<T>)
            C = max_finite<T>;
        else if (t < min_exp<T>)
            C = 0;
        else {
            C = std::exp(t);
            if (A < 0)
                C *= (int(std::floor(B + 0.5)) % 2 == 0 ? 1 : -1);
        }
    }
    /**
     * @brief Computes the gradient of an exponentiation.
     */
    constexpr static void Grad(T A, T &dA, T B, T &dB, T C, T dC) noexcept {
        if (A == 0 || std::abs(A) < epsilon<T>) {
            return;
        }
        dA += dC * (C / A);
        dB += dC * C * std::log(std::abs(A));
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
    constexpr static void Op(T A, T &C) noexcept {
        C = std::sqrt(std::max(A, T(0)));
    }
    constexpr static void Grad(T A, T &dA, T C, T dC) noexcept {
        /**
         * @brief Computes the gradient of the squareroot
         */
        dA += C < epsilon<T> ? 0 : dC / (2 * C);
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
    constexpr static void Op(T A, T &C) noexcept {
        C = std::log(std::max(A, epsilon<T>));
    }
    /**
     * @brief Computes the gradient of the logarithm
     */
    constexpr static void Grad(T A, T &dA, T C, T dC) noexcept {
        dA += dC / std::max(A, epsilon<T>);
    }
};

/**
 * @brief Unary kernel for exponent.
 * @tparam T The scalar type.
 */
template <typename T> struct safe_Exp {
    using value_type = T;

    constexpr static void Op(T A, T &C) noexcept {
        /**
         * @brief C = e^A
         */
        C = std::exp(std::max(min_exp<T>, std::min(max_exp<T>, A)));
    }
    /**
     * @brief Computes the gradient of the exp function.
     */
    constexpr static void Grad(T A, T &dA, T C, T dC) noexcept { dA += dC * C; }
};

} // namespace Kernels
} // namespace kaad
