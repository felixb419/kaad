#pragma once

#include <cstddef> // for nullptr_t
#include <math.h>
#include <limits>

namespace kaad {
    template <typename T>
    struct Kernels {

        inline static T epsilon = static_cast<T>(1000) * std::numeric_limits<T>::epsilon();
        inline static T max_exp = std::log(std::numeric_limits<T>::max());
        inline static T min_exp = std::log(std::numeric_limits<T>::min());

        struct Null {
            struct Op {};
            struct Grad {};
        };

        struct Add {
            struct Op {
                constexpr void operator()(T A, T B, T& C) const noexcept {
                    C = A + B;
                }
            };
            struct Grad {
                constexpr void operator()(T A, T& dA, T B, T& dB, T C, T dC) const noexcept {
                    dA += dC;
                    dB += dC;
                }
            };
        };

        struct Sub {
            struct Op {
                constexpr void operator()(T A, T B, T& C) const noexcept {
                    C = A - B;
                }
            };
            struct Grad {
                constexpr void operator()(T A, T& dA, T B, T& dB, T C, T dC) const noexcept {
                    dA += dC;
                    dB -= dC;
                }
            };
        };

        struct Mul {
            struct Op {
                constexpr void operator()(T A, T B, T& C) const noexcept {
                    C = A * B;
                }
            };
            struct Grad {
                constexpr void operator()(T A, T& dA, T B, T& dB, T C, T dC) const noexcept {
                    dA += dC * B; 
                    dB += dC * A;
                }
            };
        };

        struct Div {
            struct Op {
                constexpr void operator()(T A, T B, T& C) const noexcept {
                    C = A / B;
                }
            };
            struct Grad {
                constexpr void operator()(T A, T& dA, T B, T& dB, T C, T dC) const noexcept {
                    dA += dC * (1 / B);
                    dB -= dC * (A / (B * B));
                }
            };
        };

        struct Pow {
            struct Op {
                constexpr void operator()(T A, T B, T& C) const noexcept {
                    C = pow(A, B);
                }
            };
            struct Grad {
                constexpr void operator()(T A, T& dA, T B, T& dB, T C, T dC) const noexcept {
                    dA += dC * B * pow(A, B - 1);
                    dB += dC * C * log(A);
                }
            };
        };

        struct Dot {
            struct Op {
                constexpr void operator()(T A, T B, T& C) const noexcept {
                    C += A * B;
                }
            };
            struct Grad {
                constexpr void operator()(T A, T& dA, T B, T& dB, T C, T dC) const noexcept {
                    dA += dC * B;
                    dB += dC * A;
                }
            };
        };

        struct Min {
            struct Op {
                constexpr void operator()(T A, T B, T& C) const noexcept {
                    C = A < B ? A : B;
                }
            };
            struct Grad {
                constexpr void operator()(T A, T& dA, T B, T& dB, T C, T dC) const noexcept {
                    int smaller = A <= B;
                    dA += smaller ? dC : 0;
                    dB += smaller ? 0 : dC;
                }
            };
        };

        struct Max {
            struct Op {
                constexpr void operator()(T A, T B, T& C) const noexcept {
                    C = A > B ? A : B;
                }
            };
            struct Grad {
                constexpr void operator()(T A, T& dA, T B, T& dB, T C, T dC) const noexcept {
                    int bigger = A >= B;
                    dA += bigger ? dC : 0;
                    dB += bigger ? 0 : dC;
                }
            };
        };

        struct Neg {
            struct Op {
                constexpr void operator()(T A, T& C) const noexcept {
                    C = -A;
                }
            };
            struct Grad {
                constexpr void operator()(T A, T& dA, T C, T dC) const noexcept {
                    dA -= dC;
                }
            };
        };

        struct Square {
            struct Op {
                constexpr void operator()(T A, T& C) const noexcept {
                    C = A * A;
                }
            };
            struct Grad {
                constexpr void operator()(T A, T& dA, T C, T dC) const noexcept {
                    dA += dC * 2 * A;
                }
            };
        };

        struct Sqrt {
            struct Op {
                constexpr void operator()(T A, T& C) const noexcept {
#ifdef NO_STABLE_SQRT
                    C = std::sqrt(A);
#else
                    C = std::sqrt(std::max(A, 0));
#endif
                }
            };
            struct Grad {
                constexpr void operator()(T A, T& dA, T C, T dC) const noexcept {
#ifdef NO_STABLE_SQRT
                    dA += dC / (2 * C);
#else
                    dA += C < epsilon ? 0 : dC / (2 * C);
#endif
                }
            };
        };

        struct Log{
            struct Op {
                constexpr void operator()(T A, T& C) const noexcept {
#ifdef NO_STABLE_LOG
                    C = std::log(A);
#else
                    C = std::log(std::max(A, epsilon));
#endif
                }
            };
            struct Grad {
                constexpr void operator()(T A, T& dA, T C, T dC) const noexcept {
#ifdef NO_STABLE_LOG
                    dA += dC / A;
#else
                    dA += dC / std::max(A, epsilon);
#endif
                }
            };
        };

        struct Exp {
            struct Op {
                constexpr void operator()(T A, T& C) const noexcept {
#ifdef NO_STABLE_EXP
                    C = std::exp(A);
#else
                    C = std::exp(std::max(min_exp, std::min(max_exp, A)));
#endif
                }
            };
            struct Grad {
                constexpr void operator()(T A, T& dA, T C, T dC) const noexcept {
                    dA += dC * C;
                }
            };
        };

        struct Abs {
            struct Op {
                constexpr void operator()(T A, T& C) const noexcept {
                    C = std::abs(A);
                }
            };
            struct Grad {
                constexpr void operator()(T A, T& dA, T C, T dC) const noexcept {
                    dA += dC * (A > 0 ? 1 : -1);
                }
            };
        };

        struct Transp {
            struct Op {
                constexpr void operator()(T A, T& C) const noexcept {}
            };
            struct Grad {
                constexpr void operator()(T A, T& dA, T C, T dC) const noexcept {
                    dA += dC;
                }
            };
        };

        struct Sum {
            struct Op {
                constexpr void operator()(T A, T& C) const noexcept {
                    C += A;
                }
            };
            struct Grad {
                constexpr void operator()(T A, T& dA, T C, T dC) const noexcept {
                    dA += dC;
                }
            };
        };
    };

    using NullOp = class Kernels<int>::Null;
}    
