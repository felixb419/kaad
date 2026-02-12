#pragma once

#include "../scalar.hpp" // for Scalar
#include <algorithm>     // for std::max, std::min
#include <cmath>         // for std::log, std::pow, std::exp, std::sqrt
#include <cstdlib>       // for std::abs
#include <limits>        // for std::numeric_limits

namespace kaad {

/**
 * @namespace kaad::Kernels
 * @brief Contains elementary operation kernels and their corresponding
 * gradients.
 */
namespace Kernels {

struct Null {};

/**
 * @brief Binary kernel for addition.
 * @tparam T The scalar type.
 */
template <typename T> struct Add {
    using value_type = T;

    /**
     * @brief C = A + B
     */
    constexpr static void Op(T A, T B, T &C) noexcept { C = A + B; }
    /**
     * @brief Computes the gradient of an addition.
     */
    constexpr static void Grad(T A, T &dA, T B, T &dB, T C, T dC) noexcept {
        dA += dC;
        dB += dC;
    }
};

/**
 * @brief Binary kernel for subtraction.
 * @tparam T The scalar type.
 */
template <typename T> struct Sub {
    using value_type = T;

    /**
     * @brief C = A - B
     */
    constexpr static void Op(T A, T B, T &C) noexcept { C = A - B; }
    /**
     * @brief Computes the gradient of a subtraction.
     */
    constexpr static void Grad(T A, T &dA, T B, T &dB, T C, T dC) noexcept {
        dA += dC;
        dB -= dC;
    }
};

/**
 * @brief Binary kernel for multiplication.
 * @tparam T The scalar type.
 */
template <typename T> struct Mul {
    using value_type = T;

    /**
     * @brief C = A * B
     */
    constexpr static void Op(T A, T B, T &C) noexcept { C = A * B; }
    /**
     * @brief Computes the gradient a multiplication.
     */
    constexpr static void Grad(T A, T &dA, T B, T &dB, T C, T dC) noexcept {
        dA += dC * B;
        dB += dC * A;
    }
};

/**
 * @brief Binary kernel for division.
 * @tparam T The scalar type.
 */
template <typename T> struct Div {
    using value_type = T;

    /**
     * @brief C = A / B
     */
    constexpr static void Op(T A, T B, T &C) noexcept { C = A / B; }
    /**
     * @brief Computes the gradient of a division.
     */
    constexpr static void Grad(T A, T &dA, T B, T &dB, T C, T dC) noexcept {
        dA += dC * (1 / B);
        dB -= dC * (A / (B * B));
    }
};

/**
 * @brief Binary kernel for power.
 * @tparam T The scalar type.
 */
template <typename T> struct Pow {
    using value_type = T;

    /**
     * @brief C = A ^ B
     */
    constexpr static void Op(T A, T B, T &C) noexcept { C = std::pow(A, B); }
    /**
     * @brief Computes the gradient of an exponentiation.
     */
    constexpr static void Grad(T A, T &dA, T B, T &dB, T C, T dC) noexcept {
        dA += dC * B * std::pow(A, B - 1);
        dB += dC * C * std::log(A);
    }
};

/**
 * @brief Binary kernel for dot.
 * @tparam T The scalar type.
 */
template <typename T> struct Dot {
    using value_type = T;

    /**
     * @brief C += A * B
     */
    constexpr static void Op(T A, T B, T &C) noexcept { C += A * B; }
    /**
     * @brief Computes the gradient of a dot-product.
     */
    constexpr static void Grad(T A, T &dA, T B, T &dB, T C, T dC) noexcept {
        dA += dC * B;
        dB += dC * A;
    }
};

/**
 * @brief Binary kernel for minimum.
 * @tparam T The scalar type.
 */
template <typename T> struct Min {
    using value_type = T;

    /**
     * @brief C = min(A,B)
     */
    constexpr static void Op(T A, T B, T &C) noexcept { C = A < B ? A : B; }
    /**
     * @brief Computes the gradient of the min function.
     */
    constexpr static void Grad(T A, T &dA, T B, T &dB, T C, T dC) noexcept {
        int smaller = A <= B;
        dA += smaller ? dC : 0;
        dB += smaller ? 0 : dC;
    }
};

/**
 * @brief Binary kernel for maximum.
 * @tparam T The scalar type.
 */
template <typename T> struct Max {
    using value_type = T;

    /**
     * @brief C = max(A,B)
     */
    constexpr static void Op(T A, T B, T &C) noexcept { C = A > B ? A : B; }
    /**
     * @brief Computes the gradient of the max function.
     */
    constexpr static void Grad(T A, T &dA, T B, T &dB, T C, T dC) noexcept {
        int bigger = A >= B;
        dA += bigger ? dC : 0;
        dB += bigger ? 0 : dC;
    }
};

/**
 * @brief Binary kernel for no-operation.
 * @tparam T The scalar type.
 */
template <typename T> struct NoOp {
    using value_type = T;

    /**
     * @brief C = A
     */
    constexpr static void Op(T A, T &C) noexcept { C = A; }

    /**
     * @brief dA += dC
     */
    constexpr static void Grad(T A, T &dA, T C, T dC) noexcept { dA += dC; }
};

/**
 * @brief Unary kernel for summation.
 * @tparam T The scalar type.
 */
template <typename T> struct Sum {
    using value_type = T;

    /**
     * @brief C += A
     */
    constexpr static void Op(T A, T &C) noexcept { C += A; }
    /**
     * @brief Computes the gradient of the summation.
     */
    constexpr static void Grad(T A, T &dA, T C, T dC) noexcept { dA += dC; }
};

/**
 * @brief Unary kernel for negation.
 * @tparam T The scalar type.
 */
template <typename T> struct Neg {
    using value_type = T;

    /**
     * @brief C = -A
     */
    constexpr static void Op(T A, T &C) noexcept { C = -A; }
    /**
     * @brief Computes the gradient of the negation.
     */
    constexpr static void Grad(T A, T &dA, T C, T dC) noexcept { dA -= dC; }
};

/**
 * @brief Unary kernel for square.
 * @tparam T The scalar type.
 */
template <typename T> struct Square {
    using value_type = T;

    /**
     * @brief C = A ^ 2
     */
    constexpr static void Op(T A, T &C) noexcept { C = A * A; }
    /**
     * @brief Computes the gradient of the square.
     */
    constexpr static void Grad(T A, T &dA, T C, T dC) noexcept {
        dA += dC * 2 * A;
    }
};

/**
 * @brief Unary kernel for squareroot.
 * @tparam T The scalar type.
 */
template <typename T> struct Sqrt {
    using value_type = T;

    inline static T epsilon =
        static_cast<T>(1000) * std::numeric_limits<T>::epsilon();
    /**
     * @brief C = sqrt(A)
     * @note if NO_STABLE_SQRT is not defined a numerically safe version is
     * used instead.
     */
    constexpr static void Op(T A, T &C) noexcept {
#ifdef NO_STABLE_SQRT
        C = std::sqrt(A);
#else
        C = std::sqrt(std::max(A, T(0)));
#endif
    }
    constexpr static void Grad(T A, T &dA, T C, T dC) noexcept {
        /**
         * @brief Computes the gradient of the squareroot
         * @note if NO_STABLE_SQRT is not defined a numerically safe version
         * is used instead.
         */
#ifdef NO_STABLE_SQRT
        dA += dC / (2 * C);
#else
        dA += C < epsilon ? 0 : dC / (2 * C);
#endif
    }
};

/**
 * @brief Unary kernel for logarithm.
 * @tparam T The scalar type.
 */
template <typename T> struct Log {
    using value_type = T;

    inline static T epsilon =
        static_cast<T>(1000) * std::numeric_limits<T>::epsilon();
    /**
     * @brief C = log(A)
     * @note if NO_STABLE_LOG is not defined a numerically save version is
     * used instead.
     */
    constexpr static void Op(T A, T &C) noexcept {
#ifdef NO_STABLE_LOG
        C = std::log(A);
#else
        C = std::log(std::max(A, epsilon));
#endif
    }
    /**
     * @brief Computes the gradient of the logarithm
     * @note if NO_STABLE_LOG is not defined a numerically safe version is
     * used instead.
     */
    constexpr static void Grad(T A, T &dA, T C, T dC) noexcept {
#ifdef NO_STABLE_LOG
        dA += dC / A;
#else
        dA += dC / std::max(A, epsilon);
#endif
    }
};

/**
 * @brief Unary kernel for exponent.
 * @tparam T The scalar type.
 */
template <typename T> struct Exp {
    using value_type = T;

    inline static T max_exp = std::log(std::numeric_limits<T>::max());
    inline static T min_exp = std::log(std::numeric_limits<T>::min());
    constexpr static void Op(T A, T &C) noexcept {
        /**
         * @brief C = e^A
         * @if NO_STABLE_EXP is not defined a numerically safe version is
         * used instead.
         */
#ifdef NO_STABLE_EXP
        C = std::exp(A);
#else
        C = std::exp(std::max(min_exp, std::min(max_exp, A)));
#endif
    }
    /**
     * @brief Computes the gradient of the exp function.
     */
    constexpr static void Grad(T A, T &dA, T C, T dC) noexcept { dA += dC * C; }
};

/**
 * @brief Unary kernel for abs.
 * @tparam T The scalar type.
 */
template <typename T> struct Abs {
    using value_type = T;

    /**
     * @brief C = abs(A)
     */
    constexpr static void Op(T A, T &C) noexcept { C = std::abs(A); }
    /**
     * @brief Computes the gradient of the abs function.
     */
    constexpr static void Grad(T A, T &dA, T C, T dC) noexcept {
        dA += dC * (A > 0 ? 1 : -1);
    }
};

} // namespace Kernels
} // namespace kaad
