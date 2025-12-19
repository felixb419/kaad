#pragma once

#include <algorithm> // for max, min
#include <cmath>     // for log, pow, exp, sqrt
#include <cstdlib>   // for abs
#include <limits>    // for numeric_limits

namespace kaad {

/**
 * @namespace kaad::Kernels
 * @brief Contains elementary operation kernels and their corresponding
 * gradients.
 *
 * The `kaad::Kernels` namespace provides reusable, stateless structs that
 * implement basic elementwise operations (e.g., addition, subtraction) and
 * their derivatives. Each operation is represented by a pair of nested structs:
 * - `Op`: Implements the forward computation.
 * - `Grad`: Implements the backward gradient computation for automatic
 * differentiation.
 *
 * Example usage:
 * @code
 * kaad::Kernels::Add<float>::Op add;
 * float a = 1.0f, b = 2.0f, c;
 * add(a, b, c); // c == 3.0f
 * @endcode
 */
namespace Kernels {

struct Null {
    struct Op {};
    struct Grad {};
};

/**
 * @brief Binary kernel for addition.
 * @tparam T The scalar type.
 */
template <typename T> struct Add {
    struct Op {
        /**
         * @brief Performs the addition.
         * @param A First operand.
         * @param B Second operand.
         * @param C Result of A + B.
         */
        constexpr void operator()(T A, T B, T &C) const noexcept { C = A + B; }
    };
    struct Grad {
        /**
         * @brief Computes the gradient.
         * @param A   Input A (unused).
         * @param dA  Gradient accumulator for A.
         * @param B   Input B (unused).
         * @param dB  Gradient accumulator for B.
         * @param C   Output C (unused).
         * @param dC  Gradient from the output.
         */
        constexpr void operator()(T A, T &dA, T B, T &dB, T C,
                                  T dC) const noexcept {
            dA += dC;
            dB += dC;
        }
    };
};

/**
 * @brief Binary kernel for subtraction.
 * @tparam T The scalar type.
 */
template <typename T> struct Sub {
    struct Op {
        /**
         * @brief Performs the subtraction.
         * @param A First operand.
         * @param B Second operand.
         * @param C Result of A - B.
         */
        constexpr void operator()(T A, T B, T &C) const noexcept { C = A - B; }
    };
    struct Grad {
        /**
         * @brief Computes the gradient.
         * @param A   Input A (unused).
         * @param dA  Gradient accumulator for A.
         * @param B   Input B (unused).
         * @param dB  Gradient accumulator for B.
         * @param C   Output C (unused).
         * @param dC  Gradient from the output.
         */
        constexpr void operator()(T A, T &dA, T B, T &dB, T C,
                                  T dC) const noexcept {
            dA += dC;
            dB -= dC;
        }
    };
};

/**
 * @brief Binary kernel for multiplication.
 * @tparam T The scalar type.
 */
template <typename T> struct Mul {
    struct Op {
        /**
         * @brief Performs the multiplication.
         * @param A First operand.
         * @param B Second operand.
         * @param C Result of A * B.
         */
        constexpr void operator()(T A, T B, T &C) const noexcept { C = A * B; }
    };
    struct Grad {
        /**
         * @brief Computes the gradient.
         * @param A   Input A.
         * @param dA  Gradient accumulator for A.
         * @param B   Input B.
         * @param dB  Gradient accumulator for B.
         * @param C   Output C (unused).
         * @param dC  Gradient from the output.
         */
        constexpr void operator()(T A, T &dA, T B, T &dB, T C,
                                  T dC) const noexcept {
            dA += dC * B;
            dB += dC * A;
        }
    };
};

/**
 * @brief Binary kernel for division.
 * @tparam T The scalar type.
 */
template <typename T> struct Div {
    struct Op {
        /**
         * @brief Performs the division.
         * @param A First operand.
         * @param B Second operand.
         * @param C Result of A / B.
         */
        constexpr void operator()(T A, T B, T &C) const noexcept { C = A / B; }
    };
    struct Grad {
        /**
         * @brief Computes the gradient.
         * @param A   Input A.
         * @param dA  Gradient accumulator for A.
         * @param B   Input B.
         * @param dB  Gradient accumulator for B.
         * @param C   Output C (unused).
         * @param dC  Gradient from the output.
         */
        constexpr void operator()(T A, T &dA, T B, T &dB, T C,
                                  T dC) const noexcept {
            dA += dC * (1 / B);
            dB -= dC * (A / (B * B));
        }
    };
};

/**
 * @brief Binary kernel for power.
 * @tparam T The scalar type.
 */
template <typename T> struct Pow {
    struct Op {
        /**
         * @brief Computes the power.
         * @param A First operand.
         * @param B Second operand.
         * @param C Result of A ^ B.
         */
        constexpr void operator()(T A, T B, T &C) const noexcept {
            C = pow(A, B);
        }
    };
    struct Grad {
        /**
         * @brief Computes the gradient.
         * @param A   Input A.
         * @param dA  Gradient accumulator for A.
         * @param B   Input B.
         * @param dB  Gradient accumulator for B.
         * @param C   Output C.
         * @param dC  Gradient from the output.
         */
        constexpr void operator()(T A, T &dA, T B, T &dB, T C,
                                  T dC) const noexcept {
            dA += dC * B * pow(A, B - 1);
            dB += dC * C * log(A);
        }
    };
};

/**
 * @brief Binary kernel for dot.
 * @tparam T The scalar type.
 */
template <typename T> struct Dot {
    struct Op {
        /**
         * @brief Computes the dot product.
         * @param A First operand.
         * @param B Second operand.
         * @param C Result of Aᵀ × B.
         */
        constexpr void operator()(T A, T B, T &C) const noexcept { C += A * B; }
    };
    struct Grad {
        /**
         * @brief Computes the gradient.
         * @param A   Input A.
         * @param dA  Gradient accumulator for A.
         * @param B   Input B.
         * @param dB  Gradient accumulator for B.
         * @param C   Output C (unused).
         * @param dC  Gradient from the output.
         */
        constexpr void operator()(T A, T &dA, T B, T &dB, T C,
                                  T dC) const noexcept {
            dA += dC * B;
            dB += dC * A;
        }
    };
};

/**
 * @brief Binary kernel for minimum.
 * @tparam T The scalar type.
 */
template <typename T> struct Min {
    struct Op {
        /**
         * @brief Selects minimum value
         * @param A First operand.
         * @param B Second operand.
         * @param C Result of min(A, B)
         */
        constexpr void operator()(T A, T B, T &C) const noexcept {
            C = A < B ? A : B;
        }
    };
    struct Grad {
        /**
         * @brief Computes the gradient.
         * @param A   Input A.
         * @param dA  Gradient accumulator for A.
         * @param B   Input B.
         * @param dB  Gradient accumulator for B.
         * @param C   Output C (unused).
         * @param dC  Gradient from the output.
         */
        constexpr void operator()(T A, T &dA, T B, T &dB, T C,
                                  T dC) const noexcept {
            int smaller = A <= B;
            dA += smaller ? dC : 0;
            dB += smaller ? 0 : dC;
        }
    };
};

/**
 * @brief Binary kernel for maximum.
 * @tparam T The scalar type.
 */
template <typename T> struct Max {
    struct Op {
        /**
         * @brief Selects maximum value
         * @param A First operand.
         * @param B Second operand.
         * @param C Result of max(A, B)
         */
        constexpr void operator()(T A, T B, T &C) const noexcept {
            C = A > B ? A : B;
        }
    };
    struct Grad {
        /**
         * @brief Computes the gradient.
         * @param A   Input A.
         * @param dA  Gradient accumulator for A.
         * @param B   Input B.
         * @param dB  Gradient accumulator for B.
         * @param C   Output C (unused).
         * @param dC  Gradient from the output.
         */
        constexpr void operator()(T A, T &dA, T B, T &dB, T C,
                                  T dC) const noexcept {
            int bigger = A >= B;
            dA += bigger ? dC : 0;
            dB += bigger ? 0 : dC;
        }
    };
};

/**
 * @brief Unary kernel for summation.
 * @tparam T The scalar type.
 */
template <typename T> struct Sum {
    struct Op {
        /**
         * @brief Performs the summation.
         * @param A First operand.
         * @param C Result.
         */
        constexpr void operator()(T A, T &C) const noexcept { C += A; }
    };
    struct Grad {
        /**
         * @brief Computes the gradient.
         * @param A   Input A (unused).
         * @param dA  Gradient accumulator for A.
         * @param C   Output C (unused).
         * @param dC  Gradient from the output.
         */
        constexpr void operator()(T A, T &dA, T C, T dC) const noexcept {
            dA += dC;
        }
    };
};

/**
 * @brief Unary kernel for negation.
 * @tparam T The scalar type.
 */
template <typename T> struct Neg {
    struct Op {
        /**
         * @brief Performs the negation.
         * @param A First operand.
         * @param C Result of -A.
         */
        constexpr void operator()(T A, T &C) const noexcept { C = -A; }
    };
    struct Grad {
        /**
         * @brief Computes the gradient.
         * @param A   Input A (unused).
         * @param dA  Gradient accumulator for A.
         * @param C   Output C (unused).
         * @param dC  Gradient from the output.
         */
        constexpr void operator()(T A, T &dA, T C, T dC) const noexcept {
            dA -= dC;
        }
    };
};

/**
 * @brief Unary kernel for square.
 * @tparam T The scalar type.
 */
template <typename T> struct Square {
    struct Op {
        /**
         * @brief Computes the square.
         * @param A First operand.
         * @param C Result of A ^ 2.
         */
        constexpr void operator()(T A, T &C) const noexcept { C = A * A; }
    };
    struct Grad {
        /**
         * @brief Computes the gradient.
         * @param A   Input A.
         * @param dA  Gradient accumulator for A.
         * @param C   Output C (unused).
         * @param dC  Gradient from the output.
         */
        constexpr void operator()(T A, T &dA, T C, T dC) const noexcept {
            dA += dC * 2 * A;
        }
    };
};

/**
 * @brief Unary kernel for squareroot.
 * @tparam T The scalar type.
 */
template <typename T> struct Sqrt {
    inline static T epsilon =
        static_cast<T>(1000) * std::numeric_limits<T>::epsilon();
    struct Op {
        /**
         * @brief Computes the squareroot, if NO_STABLE_SQRT is not defined a
         * numerically safe version is used instead.
         * @param A First operand.
         * @param C Result of √A.
         */
        constexpr void operator()(T A, T &C) const noexcept {
#ifdef NO_STABLE_SQRT
            C = std::sqrt(A);
#else
            C = std::sqrt(std::max(A, 0));
#endif
        }
    };
    struct Grad {
        constexpr void operator()(T A, T &dA, T C, T dC) const noexcept {
            /**
             * @brief Computes the gradient, if NO_STABLE_SQRT is not defined a
             * numerically safe version is used instead.
             * @param A   Input A (unused).
             * @param dA  Gradient accumulator for A.
             * @param C   Output C.
             * @param dC  Gradient from the output.
             */
#ifdef NO_STABLE_SQRT
            dA += dC / (2 * C);
#else
            dA += C < epsilon ? 0 : dC / (2 * C);
#endif
        }
    };
};

/**
 * @brief Unary kernel for logarithm.
 * @tparam T The scalar type.
 */
template <typename T> struct Log {
    inline static T epsilon =
        static_cast<T>(1000) * std::numeric_limits<T>::epsilon();
    struct Op {
        /**
         * @brief Computes the logarithm base e of A, if NO_STABLE_LOG is not
         * defined a numerically save version is used instead.
         * @param A First operand.
         * @param C Result of ln(A).
         */
        constexpr void operator()(T A, T &C) const noexcept {
#ifdef NO_STABLE_LOG
            C = std::log(A);
#else
            C = std::log(std::max(A, epsilon));
#endif
        }
    };
    struct Grad {
        /**
         * @brief Computes the gradient, if NO_STABLE_LOG is not defined a
         * numerically safe version is used instead.
         * @param A   Input A.
         * @param dA  Gradient accumulator for A.
         * @param C   Output C (unused).
         * @param dC  Gradient from the output.
         */
        constexpr void operator()(T A, T &dA, T C, T dC) const noexcept {
#ifdef NO_STABLE_LOG
            dA += dC / A;
#else
            dA += dC / std::max(A, epsilon);
#endif
        }
    };
};

/**
 * @brief Unary kernel for exponent.
 * @tparam T The scalar type.
 */
template <typename T> struct Exp {
    inline static T max_exp = std::log(std::numeric_limits<T>::max());
    inline static T min_exp = std::log(std::numeric_limits<T>::min());
    struct Op {
        constexpr void operator()(T A, T &C) const noexcept {
            /**
             * @brief Computes e to the power of A, if NO_STABLE_EXP is not
             * defined a numerically safe version is used instead.
             * @param A First operand.
             * @param C Result of e ^ A.
             */
#ifdef NO_STABLE_EXP
            C = std::exp(A);
#else
            C = std::exp(std::max(min_exp, std::min(max_exp, A)));
#endif
        }
    };
    struct Grad {
        /**
         * @brief Computes the gradient.
         * @param A   Input A (unused).
         * @param dA  Gradient accumulator for A.
         * @param C   Output C.
         * @param dC  Gradient from the output.
         */
        constexpr void operator()(T A, T &dA, T C, T dC) const noexcept {
            dA += dC * C;
        }
    };
};

/**
 * @brief Unary kernel for abs.
 * @tparam T The scalar type.
 */
template <typename T> struct Abs {
    struct Op {
        /**
         * @brief Computes the absolute value.
         * @param A First operand.
         * @param C Result of |A|.
         */
        constexpr void operator()(T A, T &C) const noexcept { C = std::abs(A); }
    };
    struct Grad {
        /**
         * @brief Computes the gradient.
         * @param A   Input A.
         * @param dA  Gradient accumulator for A.
         * @param C   Output C (unused).
         * @param dC  Gradient from the output.
         */
        constexpr void operator()(T A, T &dA, T C, T dC) const noexcept {
            dA += dC * (A > 0 ? 1 : -1);
        }
    };
};

} // namespace Kernels
} // namespace kaad
