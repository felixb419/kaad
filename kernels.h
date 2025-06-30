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
};

using NullOp = class Kernels<nullptr_t>::Null;