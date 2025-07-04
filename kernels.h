#include <cstddef> // for nullptr_t

#pragma once

template <typename T>
struct Kernels {
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
                C = sqrt(A);
            }
        };
        struct Grad {
            constexpr void operator()(T A, T& dA, T C, T dC) const noexcept {
                dA += dC / (2 * C);
            }
        };
    };

    struct Log{
        struct Op {
            constexpr void operator()(T A, T& C) const noexcept {
                C = log(A);
            }
        };
        struct Grad {
            constexpr void operator()(T A, T& dA, T C, T dC) const noexcept {
                dA += dC / A;
            }
        };
    };

    struct Exp {
        struct Op {
            constexpr void operator()(T A, T& C) const noexcept {
                C = exp(A);
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
                C = abs(A);
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

using NullOp = class Kernels<std::nullptr_t>::Null;
