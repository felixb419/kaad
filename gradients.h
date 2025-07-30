#pragma once

#include "operations.h" // for batch_matmul, matmul
#include <cstddef>      // for size_t

namespace kaad {
template <typename T, class Grad>
using unaryGrad = void (*)(const T *A, T *dA, const T *C, const T *dC,
                           const T *end, Grad grad);
template <typename T, class Grad>
using binaryGrad = void (*)(const T *A, T *dA, const T *B, T *dB, const T *C,
                            const T *dC, const T *end, Grad grad);
template <typename T, class Grad>
using flexBinaryGrad = void (*)(const T *A, T *dA, const T *B, T *dB,
                                const T *C, const T *dC, int *strideA,
                                int *strideB, int *strideC, size_t *c_offset,
                                int N, Grad grad);
template <typename T>
using matmulGrad = void (*)(const T *A, T *dA, const T *B, T *dB, const T *C,
                            const T *dC, int *a_dim, int *b_dim, int *k,
                            int *strideA, int *strideB, int *strideC);
template <typename T>
using batchmatmulGrad = void (*)(const T *A, T *dA, const T *B, T *dB,
                                 const T *C, const T *dC, int **strideA,
                                 int **strideB, int **strideC, int **c_shape,
                                 int *a_off, int *b_off, int *k, int N);
template <typename T>
using sumDimGrad = void (*)(T *dA, const T *dC, int *strideA, int *strideC,
                            size_t *a_offset, int N);
template <typename T>
using meanGrad = void (*)(T *dA, const T *dC, const T *dA_end, T divisor);
template <typename T>
using meanDimGrad = void (*)(const T *A, T *dA, const T *C, T *dC, int *strideA,
                             int *strideC, size_t *a_offset, int N, T divisor,
                             T *c_end);
template <typename T>
using sliceGrad = void (*)(T *dA, const T *dC, int *strideA, int *strideC,
                           size_t *start_offset, size_t *c_offset, int N);

namespace Gradients {

// d/dx[ f(g(x,...)) ] = f'(g(x,...)) * g'(x,...)

namespace binary {

template <typename T, class Grad>
void scalarRhs(const T *A, T *dA, const T *B, T *dB, const T *C, const T *dC,
               const T *C_end, Grad grad) {
    for (; C != C_end; A++, dA++, C++, dC++) {
        grad(*A, *dA, *B, *dB, *C, *dC);
    }
}
template <typename T, class Grad>
void scalarLhs(const T *A, T *dA, const T *B, T *dB, const T *C, const T *dC,
               const T *C_end, Grad grad) {
    for (; C != C_end; B++, dB++, C++, dC++) {
        grad(*A, *dA, *B, *dB, *C, *dC);
    }
}
template <typename T, class Grad>
void pointwise(const T *A, T *dA, const T *B, T *dB, const T *C, const T *dC,
               const T *C_end, Grad grad) {
    for (; C != C_end; A++, dA++, B++, dB++, C++, dC++) {
        grad(*A, *dA, *B, *dB, *C, *dC);
    }
}
template <typename T, class Grad>
void flexible(const T *A, T *dA, const T *B, T *dB, const T *C, const T *dC,
              int *strideA, int *strideB, int *strideC, size_t *c_offset, int N,
              Grad grad) {
    const T *end = C + *c_offset;
    if (N <= 1) {
        for (; C != end; A += *strideA, B += *strideB, C += *strideC,
                         dA += *strideA, dB += *strideB, dC += *strideC) {
            grad(*A, *dA, *B, *dB, *C, *dC);
        }
    } else {
        for (; C < end; A += *strideA, B += *strideB, C += *strideC,
                        dA += *strideA, dB += *strideB, dC += *strideC) {
            flexible(A, dA, B, dB, C, dC, strideA + 1, strideB + 1, strideC + 1,
                     c_offset + 1, N - 1, grad);
        }
    }
}

template <typename T, class Grad, int N>
void flexible(const T *A, T *dA, const T *B, T *dB, const T *C, const T *dC,
              int *strideA, int *strideB, int *strideC, size_t *c_offset, int _,
              Grad grad) {
    const T *end = C + *c_offset;
    if constexpr (N <= 1) {
        for (; C != end; A += *strideA, B += *strideB, C += *strideC,
                         dA += *strideA, dB += *strideB, dC += *strideC) {
            grad(*A, *dA, *B, *dB, *C, *dC);
        }
    } else {
        for (; C != end; A += *strideA, B += *strideB, C += *strideC,
                         dA += *strideA, dB += *strideB, dC += *strideC) {
            flexible<T, Grad, N - 1>(A, dA, B, dB, C, dC, strideA + 1,
                                     strideB + 1, strideC + 1, c_offset + 1, 0,
                                     grad);
        }
    }
}

// f(A,B) = A dot B
// df/dA = B
// df/dB = A
template <typename T, class Grad>
void dot(const T *A, T *dA, const T *B, T *dB, const T *C, const T *dC,
         const T *A_end, Grad _) {
    for (; A != A_end; A++, dA++, B++, dB++) {
        *dA += *dC * (*B);
        *dB += *dC * (*A);
    }
}
template <typename T, class Grad>
void scalarDot(const T *A, T *dA, const T *B, T *dB, const T *C, const T *dC,
               const T *A_end, Grad _) {
    for (; A != A_end; A++, dA++) {
        *dA += *dC * (*B);
        *dB += *dC * (*A);
    }
}

// f(A,B) = AB
// dC/dA = B^T
// dC/dB = A^T
template <typename T>
void matmul(const T *A, T *dA, const T *B, T *dB, const T *C, const T *dC,
            int *a_dim, int *b_dim, int *k, int *strideA, int *strideB,
            int *strideC) {
    // dA = dC * B^T
    Operations::binary::matmul(dC, B, dA, a_dim[0], b_dim[0], k[0], strideC,
                               strideB, strideA);
    // dB = A^T * dC
    Operations::binary::matmul(A, dC, dB, a_dim[1], b_dim[1], k[1], strideA + 2,
                               strideC + 2, strideB + 2);
}

template <typename T>
void batch_matmul(const T *A, T *dA, const T *B, T *dB, const T *C, const T *dC,
                  int **strideA, int **strideB, int **strideC, int **c_shape,
                  int *a_off, int *b_off, int *k, int N) {
    // dA = dC * B^T
    Operations::binary::batch_matmul<T>(dC, B, dA, strideC[0], strideB[0],
                                        strideA[0], c_shape[0], a_off[0],
                                        b_off[0], k[0], N);
    // dB = A^T * dC
    Operations::binary::batch_matmul<T>(A, dC, dB, strideA[1], strideC[1],
                                        strideB[1], c_shape[1], a_off[1],
                                        b_off[1], k[1], N);
}

template <typename T, int N>
void batch_matmul(const T *A, T *dA, const T *B, T *dB, const T *C, const T *dC,
                  int **strideA, int **strideB, int **strideC, int **c_shape,
                  int *a_off, int *b_off, int *k, int _) {
    // dA = dC * B^T
    Operations::binary::batch_matmul<T, N>(dC, B, dA, strideC[0], strideB[0],
                                           strideA[0], c_shape[0], a_off[0],
                                           b_off[0], k[0], 0);
    // dB = A^T * dC
    Operations::binary::batch_matmul<T, N>(A, dC, dB, strideA[1], strideC[1],
                                           strideB[1], c_shape[1], a_off[1],
                                           b_off[1], k[1], 0);
}

} // namespace binary

namespace unary {

template <typename T, class Grad>
void pointwise(const T *A, T *dA, const T *C, const T *dC, const T *C_end,
               Grad grad) {
    for (; C != C_end; A++, dA++, C++, dC++) {
        grad(*A, *dA, *C, *dC);
    }
}

template <typename T, class Grad>
void scalarRhs(const T *A, T *dA, const T *C, const T *dC, const T *A_end,
               Grad grad) {
    for (; A != A_end; A++, dA++) {
        grad(*A, *dA, *C, *dC);
    }
}

template <typename T, class Grad>
void unary_flexible(const T *A, T *dA, const T *C, const T *dC, int *strideA,
                    int *strideC, int *reps, int *count, size_t D, Grad grad) {
    int indA = 0, indC = 0;
    while (1) {

        grad(A[indA], dA[indA], C[indC], dC[indC]);

        for (int dim = D - 1; dim >= 0; dim--) {
            count[dim]--;
            if (count[dim] >= 0) {
                indA += strideA[dim];
                indC += strideC[dim];
                break;
            }

            count[dim] = reps[dim];
            if (dim == 0)
                goto end;
        }
    }
end:;
}

// f(A) = A^T
// df/dA = 1
template <typename T, class Grad>
void transp(const T *A, T *dA, const T *C, const T *dC, size_t len) {
    const T *end = C + len;
    for (; C != end; dA++, dC++) {
        *dA += *dC;
    }
}

// f(A) = sum(A)
// df_dA = tensor with shape of A filled with 1
template <typename T>
void sum_dim(T *dA, const T *dC, int *strideA, int *strideC, size_t *a_offset,
             int N) {
    const T *end = dA + *a_offset;
    if (N <= 1) {
        for (; dA != end; dA += *strideA, dC += *strideC) {
            *dA += *dC;
        }
    } else {
        for (; dA != end; dA += *strideA, dC += *strideC) {
            sum_dim(dA, dC, strideA + 1, strideC + 1, a_offset + 1, N - 1);
        }
    }
}

template <typename T, int N>
void sum_dim(T *dA, const T *dC, int *strideA, int *strideC, size_t *a_offset,
             int _) {
    const T *end = dA + *a_offset;
    if constexpr (N <= 1) {
        for (; dA != end; dA += *strideA, dC += *strideC) {
            *dA += *dC;
        }
    } else {
        for (; dA != end; dA += *strideA, dC += *strideC) {
            sum_dim<T, N - 1>(dA, dC, strideA + 1, strideC + 1, a_offset + 1,
                              0);
        }
    }
}

// f(A) = mean(A)
// df_dA = tensor with shape of A filled with 1 / (len of A)
template <typename T>
void mean(T *dA, const T *dC, const T *dA_end, T divisor) {
    T mean = *dC / divisor;
    for (; dA != dA_end; dA++) {
        *dA += mean;
    }
}

template <typename T>
void mean_dim_impl(const T *A, T *dA, const T *C, T *dC, int *strideA,
                   int *strideC, size_t *a_offset, int N) {
    const T *end = dA + *a_offset;
    if (N <= 1) {
        for (; dA != end; dA += *strideA, dC += *strideC) {
            *dA += *dC;
        }
    } else {
        for (; dA != end; dA += *strideA, dC += *strideC) {
            mean_dim_impl(A, dA, C, dC, strideA + 1, strideC + 1, a_offset + 1,
                          N - 1);
        }
    }
}
template <typename T>
void mean_dim(const T *A, T *dA, const T *C, T *dC, int *strideA, int *strideC,
              size_t *a_offset, int N, T divisor, T *dA_end) {
    mean_dim_impl(A, dA, C, dC, strideA, strideC, a_offset, N);
    for (; dA != dA_end; dA++) {
        *dA /= divisor;
    }
}

template <typename T, int N>
void mean_dim_impl(const T *A, T *dA, const T *C, T *dC, int *strideA,
                   int *strideC, size_t *a_offset, int _) {
    const T *end = dA + *a_offset;
    if constexpr (N <= 1) {
        for (; dA != end; dA += *strideA, dC += *strideC) {
            *dA += *dC;
        }
    } else {
        for (; dA != end; dA += *strideA, dC += *strideC) {
            mean_dim_impl<T, N - 1>(A, dA, C, dC, strideA + 1, strideC + 1,
                                    a_offset + 1, 0);
        }
    }
}
template <typename T, int N>
void mean_dim(const T *A, T *dA, const T *C, T *dC, int *strideA, int *strideC,
              size_t *a_offset, int _, T divisor, T *dA_end) {
    mean_dim_impl<T, N>(A, dA, C, dC, strideA, strideC, a_offset, 0);
    for (; dA != dA_end; dA++) {
        *dA /= divisor;
    }
}

template <typename T>
void slice(T *dA, const T *dC, int *strideA, int *strideC, size_t *start_offset,
           size_t *c_offset, int N) {
    dA += *start_offset;
    const T *end = dC + *c_offset;
    if (N <= 1) {
        for (; dC != end; dA += *strideA, dC += *strideC) {
            *dA = *dC;
        }
    } else {
        for (; dC < end; dA += *strideA, dC += *strideC) {
            slice(dA, dC, strideA + 1, strideC + 1, start_offset + 1,
                  c_offset + 1, N - 1);
        }
    }
}

template <typename T, int N>
void slice(T *dA, const T *dC, int *strideA, int *strideC, size_t *start_offset,
           size_t *c_offset, int _) {
    dA += *start_offset;
    const T *end = dC + *c_offset;
    if constexpr (N <= 1) {
        for (; dC != end; dA += *strideA, dC += *strideC) {
            *dA = *dC;
        }
    } else {
        for (; dC < end; dA += *strideA, dC += *strideC) {
            slice<T, N - 1>(dA, dC, strideA + 1, strideC + 1, start_offset + 1,
                            c_offset + 1, 0);
        }
    }
}

} // namespace unary

} // namespace Gradients
} // namespace kaad
