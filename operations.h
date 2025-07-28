#pragma once

#include <algorithm> // for std::copy
#include <stddef.h>  // for size_t

namespace kaad {
template <typename T, class Op>
using unaryOp = void (*)(const T *A, T *C, size_t len, Op op);
template <typename T, class Op>
using binaryOp = void (*)(const T *A, const T *B, T *C, size_t len, Op op);
template <typename T, class Op>
using flexBinaryOp = void (*)(const T *A, const T *B, T *C, int *strideA,
                              int *strideB, int *strideC, size_t *c_offset,
                              int N, Op op);
template <typename T>
using matmulOp = void (*)(const T *A, const T *B, T *C, int a_dim, int b_dim,
                          int k, int *strideA, int *strideB, int *strideC);
template <typename T>
using batchmatmulOp = void (*)(const T *A, const T *B, T *C, int *strideA,
                               int *strideB, int *strideC, int *c_shape,
                               int a_off, int b_off, int k, int N);
template <typename T>
using sumDimOp = void (*)(const T *A, T *C, int *strideA, int *strideC,
                          size_t *a_offset, int N);
template <typename T>
using meanDimOp = void (*)(const T *A, T *C, int *strideA, int *strideC,
                           size_t *a_offset, int N, T divisor, T *c_end);

namespace Operations {
/*
BINARY OPS
*/

// perform op so that: C[m,n,...] = op( A[m,n,...], B[0] )
// shapes of C and A must be the same, shape of B must be (1)
template <typename T, class Op>
void scalarRhs(const T *A, const T *B, T *C, size_t len, Op op) {
    T *end = C + len;
    for (; C != end; A++, C++) {
        op(*A, *B, *C);
    }
}

// perform op so that: C[m,n,...] = op( A[0], B[m,n,...])
// shapes of out and tensor must be the same, shape of scalar must be (1)
template <typename T, class Op>
void scalarLhs(const T *A, const T *B, T *C, size_t len, Op op) {
    T *end = C + len;
    for (; C != end; B++, C++) {
        op(*A, *B, *C);
    }
}

// perform op so that so that: C[m,n,...] = op( A[m,n,...], B[m,n...] )
// shape of all operands must be indentical
template <typename T, class Op>
void pointwise(const T *A, const T *B, T *C, size_t len, Op op) {
    T *end = C + len;
    for (; C != end; A++, B++, C++) {
        op(*A, *B, *C);
    }
}

// perform op flexible so that: C = op( A, B )
// shape of C must be a valid broadcast of A and B
template <typename T, class Op>
void flexible(const T *A, const T *B, T *C, int *strideA, int *strideB,
              int *strideC, size_t *c_offset, int N, Op op) {
    const T *end = C + *c_offset;
    if (N <= 1) {
        for (; C != end; A += *strideA, B += *strideB, C += *strideC) {
            op(*A, *B, *C);
        }
    } else {
        for (; C < end; A += *strideA, B += *strideB, C += *strideC) {
            flexible(A, B, C, strideA + 1, strideB + 1, strideC + 1,
                     c_offset + 1, N - 1, op);
        }
    }
}

template <typename T, class Op, int N>
void flexible(const T *A, const T *B, T *C, int *strideA, int *strideB,
              int *strideC, int *c_offset, int _, Op op) {
    const T *end = C + *c_offset;
    if constexpr (N <= 1) {
        for (; C != end; A += *strideA, B += *strideB, C += *strideC) {
            op(*A, *B, *C);
        }
    } else {
        for (; C < end; A += *strideA, B += *strideB, C += *strideC) {
            flexible<T, Op, N - 1>(A, B, C, strideA + 1, strideB + 1,
                                   strideC + 1, c_offset + 1, 0, op);
        }
    }
}

// compute do product of A and B into C
// A must be 1d vector, B and C must be scalar
template <typename T> void scalarDot(const T *A, const T *B, T *C, size_t len) {
    T *end = A + len;
    for (; A != end; A++) {
        *C += *A * (*B);
    }
}
// compute do product of A and B into C
// A and B must be 1d vectors of same length, C must be scalar
template <typename T> void dot(const T *A, const T *B, T *C, size_t len) {
    const T *end = A + len;
    for (; A != end; A++, B++) {
        *C += *A * (*B);
    }
}

// matrix multiply A and B so that C = AB
// A and B must be 2d and width of A is equalt to height of B
template <typename T>
void matmul(const T *A, const T *B, T *C, int a_dim, int b_dim, int k,
            int *strideA, int *strideB, int *strideC) {
    const T *_A;
    const T *_B;
    const T *__B;
    for (int a_idx = 0; a_idx < a_dim;
         a_idx++, A += strideA[0], C += strideC[0]) {
        _B = B;
        for (int b_idx = 0; b_idx < b_dim;
             b_idx++, _B += strideB[1], C += strideC[1]) {
            _A = A;
            __B = _B;
            for (int i = 0; i < k; i++, _A += strideA[1], __B += strideB[0]) {
                *C += (*_A) * (*__B);
            }
        }
    }
}

// matrix multiply A and B so that C = AB
// last two dimensions of A and B must me matrix multipliable
// all dimensions higher than 2 are regarded as batch dimensions
template <typename T>
void batch_matmul(const T *A, const T *B, T *C, int *strideA, int *strideB,
                  int *strideC, int *c_shape, int a_off, int b_off, int k,
                  int N) {
    const T *end = C + (*c_shape) * (*strideC);
    if (N <= 1) {
        for (int i = 0; i < *c_shape;
             i++, A += *strideA, B += *strideB, C += *strideC) {
            const T *_A = A, *_B = B;
            for (int j = 0; j < k; j++, _A += a_off, _B += b_off) {
                *C += (*_A) * (*_B);
            }
        }
    } else {
        for (int i = 0; i < *c_shape;
             i++, A += *strideA, B += *strideB, C += *strideC) {
            batch_matmul(A, B, C, strideA + 1, strideB + 1, strideC + 1,
                         c_shape + 1, a_off, b_off, k, N - 1);
        }
    }
}

template <typename T, int N>
void batch_matmul(const T *A, const T *B, T *C, int *strideA, int *strideB,
                  int *strideC, int *c_shape, int a_off, int b_off, int k,
                  int _) {
    const T *end = C + (*c_shape) * (*strideC);
    if constexpr (N <= 1) {
        for (int i = 0; i < *c_shape;
             i++, A += *strideA, B += *strideB, C += *strideC) {
            const T *_A = A, *_B = B;
            for (int j = 0; j < k; j++, _A += a_off, _B += b_off) {
                *C += (*_A) * (*_B);
            }
        }
    } else {
        for (int i = 0; i < *c_shape;
             i++, A += *strideA, B += *strideB, C += *strideC) {
            batch_matmul<T, N - 1>(A, B, C, strideA + 1, strideB + 1,
                                   strideC + 1, c_shape + 1, a_off, b_off, k,
                                   0);
        }
    }
}

/*
UNARY OPS
*/

template <typename T, class Op>
void unary_pointwise(const T *A, T *C, size_t len, Op op) {
    T *end = C + len;
    for (; C != end; A++, C++) {
        op(*A, *C);
    }
}

template <typename T, class Op>
void unary_scalarRhs(const T *A, T *C, size_t len, Op op) {
    const T *end = A + len;
    for (; A != end; A++) {
        op(*A, *C);
    }
}

// transposing doesnt change the value array so A gets copied to C
template <typename T, class Op>
void transpose(const T *A, T *C, size_t len, Op op) {
    std::copy(A, A + len, C);
}

// computes sum of tensor along dimension
// out must be same shape as A with one dimension missing
template <typename T>
void sum_dim(const T *A, T *C, int *strideA, int *strideC, size_t *a_offset,
             int N) {
    const T *end = A + *a_offset;
    if (N <= 1) {
        for (; A != end; A += *strideA, C += *strideC) {
            *C += *A;
        }
    } else {
        for (; A != end; A += *strideA, C += *strideC) {
            sum_dim(A, C, strideA + 1, strideC + 1, a_offset + 1, N - 1);
        }
    }
}

template <typename T, int N>
void sum_dim(const T *A, T *C, int *strideA, int *strideC, size_t *a_offset,
             int _) {
    const T *end = A + *a_offset;
    if constexpr (N <= 1) {
        for (; A != end; A += *strideA, C += *strideC) {
            *C += *A;
        }
    } else {
        for (; A != end; A += *strideA, C += *strideC) {
            sum_dim<T, N - 1>(A, C, strideA + 1, strideC + 1, a_offset + 1, 0);
        }
    }
}

// saves mean of A into out
// B has to be a scalar
template <typename T, class Op> void mean(const T *A, T *C, size_t len, Op _) {
    const T *end = A + len;
    for (; A != end; A++) {
        *C += *A;
    }
    *C /= len;
}

// computes mean of tensor along dimension
// out must be same shape as A with one dimension missing
template <typename T>
void mean_dim_impl(const T *A, T *C, int *strideA, int *strideC,
                   size_t *a_offset, int N) {
    const T *end = A + *a_offset;
    if (N <= 1) {
        for (; A != end; A += *strideA, C += *strideC) {
            *C += *A;
        }
    } else {
        for (; A != end; A += *strideA, C += *strideC) {
            mean_dim_impl(A, C, strideA + 1, strideC + 1, a_offset + 1, N - 1);
        }
    }
}
template <typename T>
void mean_dim(const T *A, T *C, int *strideA, int *strideC, size_t *a_offset,
              int N, T divisor, T *c_end) {
    mean_dim_impl(A, C, strideA, strideC, a_offset, N);
    for (; C != c_end; C++) {
        *C /= divisor;
    }
}

template <typename T, int N>
void mean_dim_impl(const T *A, T *C, int *strideA, int *strideC,
                   size_t *a_offset, int _) {
    const T *end = A + *a_offset;
    if constexpr (N <= 1) {
        for (; A != end; A += *strideA, C += *strideC) {
            *C += *A;
        }
    } else {
        for (; A != end; A += *strideA, C += *strideC) {
            mean_dim_impl<T, N - 1>(A, C, strideA + 1, strideC + 1,
                                    a_offset + 1, 0);
        }
    }
}
template <typename T, int N>
void mean_dim(const T *A, T *C, int *strideA, int *strideC, size_t *a_offset,
              int _, T divisor, T *c_end) {
    mean_dim_impl<T, N>(A, C, strideA, strideC, a_offset, 0);
    for (; C != c_end; C++) {
        *C /= divisor;
    }
}

} // namespace Operations
} // namespace kaad
