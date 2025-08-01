#pragma once

#include <algorithm> // for std::copy
#include <stddef.h>  // for size_t

namespace kaad {
template <typename T, class Op>
using unaryOp = void (*)(const T *A, T *C, const T *C_end, Op op);
template <typename T, class Op>
using binaryOp = void (*)(const T *A, const T *B, T *C, const T *C_end, Op op);
template <typename T, class Op>
using flexBinaryOp = void (*)(const T *A, const T *B, T *C, int *strideA,
                              int *strideB, int *strideC, size_t *C_offset,
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
                          size_t *A_offset, int N);
template <typename T>
using meanOp = void (*)(const T *A, T *C, const T *A_end, T divisor);
template <typename T>
using meanDimOp = void (*)(const T *A, T *C, int *strideA, int *strideC,
                           size_t *A_offset, int N, T divisor, T *C_end);
template <typename T>
using sliceOp = void (*)(const T *A, T *C, int *strideA, int *strideC,
                         size_t *start_offset_a, size_t *C_offset, int N);

/**
 * @namespace kaad::Operations
 * @brief Contains elementwise binary and unary operation implementations for
 * various shape patterns, including scalar, pointwise, and broadcasted tensor
 * operations.
 */
namespace Operations {

/**
 * @namespace kaad::Operations::binary
 * @brief Contains all binary operations
 */
namespace binary {

/**
 * @brief Applies a binary operation where the right-hand side is a scalar.
 *
 * Computes: C[i] = op(A[i], B[0])
 *
 * @tparam T Element type
 * @tparam Op A callable binary operation.
 *
 * @param A Pointer to the scalar input (shape = 1).
 * @param B Pointer to the start of input tensor B.
 * @param C Pointer to the output tensor.
 * @param C_end Pointer to the end of output tensor.
 * @param op The binary operation to apply.
 */
template <typename T, class Op>
void scalarRhs(const T *A, const T *B, T *C, const T *C_end, Op op) {
    for (; C != C_end; A++, C++) {
        op(*A, *B, *C);
    }
}

/**
 * @brief Applies a binary operation where the left-hand side is a scalar.
 *
 * Computes: C[i] = op(A[0], B[i])
 *
 * @tparam T Element type
 * @tparam Op A callable binary operation.
 *
 * @param A Pointer to the start of input tensor A.
 * @param B Pointer to the start of input tensor B.
 * @param C Pointer to the output tensor.
 * @param C_end Pointer to the end of output tensor.
 * @param op The binary operation to apply.
 */
template <typename T, class Op>
void scalarLhs(const T *A, const T *B, T *C, const T *C_end, Op op) {
    for (; C != C_end; B++, C++) {
        op(*A, *B, *C);
    }
}

/**
 * @brief Applies a binary operation on two tensors with identical shapes and
 * strides.
 *
 * Computes: C[i] = op(A[i], B[i])
 *
 * @tparam T Element type
 * @tparam Op A callable binary operation.
 *
 * @param A Pointer to the start of input tensor A.
 * @param B Pointer to the start of input tensor B.
 * @param C Pointer to the start of output tensor C.
 * @param C_end Pointer to the end of output tensor.
 * @param op The binary operation to apply.
 */
template <typename T, class Op>
void pointwise(const T *A, const T *B, T *C, const T *C_end, Op op) {
    for (; C != C_end; A++, B++, C++) {
        op(*A, *B, *C);
    }
}

/**
 * @brief Applies a binary operation with broadcasting support.
 *
 * Computes: C = op(A, B), using broadcasting over N dimensions.
 * This version uses runtime recursion.
 *
 * @tparam T Element type
 * @tparam Op A callable binary operation.
 *
 * @param A Pointer to the start of input tensor A.
 * @param B Pointer to the start of input tensor B.
 * @param C Pointer to the start of output tensor C.
 * @param strideA Stride array for A.
 * @param strideB Stride array for B.
 * @param strideC Stride array for C.
 * @param C_offset Total number of elements and per-dim offsets of C.
 * @param N Number of dimensions.
 * @param op The binary operation to apply.
 */
template <typename T, class Op>
void flexible(const T *A, const T *B, T *C, int *strideA, int *strideB,
              int *strideC, size_t *C_offset, int N, Op op) {
    const T *end = C + *C_offset;
    if (N <= 1) {
        for (; C != end; A += *strideA, B += *strideB, C += *strideC) {
            op(*A, *B, *C);
        }
    } else {
        for (; C < end; A += *strideA, B += *strideB, C += *strideC) {
            flexible(A, B, C, strideA + 1, strideB + 1, strideC + 1,
                     C_offset + 1, N - 1, op);
        }
    }
}

/**
 * @brief Compile-time recursive version of flexible().
 * @see flexible(const T*, const T*, T*, int*, int*, int*, size_t*, int, Op)
 */
template <typename T, class Op, int N>
void flexible(const T *A, const T *B, T *C, int *strideA, int *strideB,
              int *strideC, size_t *C_offset, int _, Op op) {
    const T *end = C + *C_offset;
    if constexpr (N <= 1) {
        for (; C != end; A += *strideA, B += *strideB, C += *strideC) {
            op(*A, *B, *C);
        }
    } else {
        for (; C < end; A += *strideA, B += *strideB, C += *strideC) {
            flexible<T, Op, N - 1>(A, B, C, strideA + 1, strideB + 1,
                                   strideC + 1, C_offset + 1, 0, op);
        }
    }
}

/**
 * @brief Computes the scalar dot product: C += Aᵀ × B
 *
 * A is a 1D vector, B and C are scalars. Result is accumulated into *C.
 *
 * @tparam T Value type.
 * @tparam Op Dummy type for consistency.
 *
 * @param A Pointer to the beginning of vector A.
 * @param B Pointer to scalar B.
 * @param C Pointer to scalar result C.
 * @param A_end Pointer to the end of vector A.
 * @param _ Ignored operation type.
 */
template <typename T, class Op>
void scalarDot(const T *A, const T *B, T *C, const T *A_end, Op _) {
    for (; A != A_end; A++) {
        *C += *A * (*B);
    }
}

/**
 * @brief Computes the dot product of two 1D vectors: C += Aᵀ × B
 *
 * A and B must be the same length. Result is accumulated into *C.
 *
 * @tparam T Value type.
 * @tparam Op Dummy type for consistency.
 *
 * @param A Pointer to the beginning of vector A.
 * @param B Pointer to scalar B.
 * @param B Pointer to vector B.
 * @param C Pointer to scalar result C.
 * @param A_end Pointer to the end of vector A.
 * @param _ Ignored operation type.
 */
template <typename T, class Op>
void dot(const T *A, const T *B, T *C, const T *A_end, Op _) {
    for (; A != A_end; A++, B++) {
        *C += *A * (*B);
    }
}

/**
 * @brief Performs matrix multiplication: C = A × B
 *
 * A and B must be 2D matrices, with shape constraints: A.shape[1] ==
 * B.shape[0].
 *
 * @tparam T Value type.
 *
 * @param A Pointer to matrix A.
 * @param B Pointer to matrix B.
 * @param C Pointer to result matrix C.
 * @param a_dim Rows of A.
 * @param b_dim Columns of B.
 * @param k Shared inner dimension (A.cols == B.rows).
 * @param strideA Strides for matrix A.
 * @param strideB Strides for matrix B.
 * @param strideC Strides for matrix C.
 */
template <typename T>
void matmul(const T *A, const T *B, T *C, int a_dim, int b_dim, int k,
            int *strideA, int *strideB, int *strideC) {
    const T *rowA;
    const T *colB;
    const T *elemB;
    for (int a_idx = 0; a_idx < a_dim;
         a_idx++, A += strideA[0], C += strideC[0]) {
        colB = B;
        for (int b_idx = 0; b_idx < b_dim;
             b_idx++, colB += strideB[1], C += strideC[1]) {
            rowA = A;
            elemB = colB;
            for (int i = 0; i < k;
                 i++, rowA += strideA[1], elemB += strideB[0]) {
                *C += (*rowA) * (*elemB);
            }
        }
    }
}

/**
 * @brief Performs batched matrix multiplication: C = A × B
 *
 * Multiplies last two dimensions of A and B like matrices; other dimensions are
 * batched. Recurses over higher dimensions dynamically.
 *
 * @tparam T Value type.
 *
 * @param A Pointer to the start of input tensor A.
 * @param B Pointer to the start of input tensor B.
 * @param C Pointer to the start of output tensor C.
 * @param strideA Strides of A.
 * @param strideB Strides of B.
 * @param strideC Strides of C.
 * @param c_shape Shape of output tensor C.
 * @param a_off Offset between A matrix rows.
 * @param b_off Offset between B matrix columns.
 * @param k Shared matrix inner dimension.
 * @param N Number of dimensions.
 */
template <typename T>
void batch_matmul(const T *A, const T *B, T *C, int *strideA, int *strideB,
                  int *strideC, int *c_shape, int a_off, int b_off, int k,
                  int N) {
    const T *end = C + (*c_shape) * (*strideC);
    if (N <= 1) {
        for (int i = 0; i < *c_shape;
             i++, A += *strideA, B += *strideB, C += *strideC) {
            const T *rowA = A, *colB = B;
            for (int j = 0; j < k; j++, rowA += a_off, colB += b_off) {
                *C += (*rowA) * (*colB);
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

/**
 * @brief Compile-time recursive version of batched matrix multiplication.
 * @see batch_matmul(const T*, const T*, T*, int*, int*, int*, int*, int, int,
 * int, int)
 */
template <typename T, int N>
void batch_matmul(const T *A, const T *B, T *C, int *strideA, int *strideB,
                  int *strideC, int *c_shape, int a_off, int b_off, int k,
                  int _) {
    const T *end = C + (*c_shape) * (*strideC);
    if constexpr (N <= 1) {
        for (int i = 0; i < *c_shape;
             i++, A += *strideA, B += *strideB, C += *strideC) {
            const T *rowA = A, *colB = B;
            for (int j = 0; j < k; j++, rowA += a_off, colB += b_off) {
                *C += (*rowA) * (*colB);
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

} // namespace binary

/**
 * @namespace kaad::Operations::unary
 * @brief Contains all unary operations
 */
namespace unary {

/**
 * @brief Applies a unary operation to each element of A and writes to C.
 * @tparam T Element type
 * @tparam Op Unary operation type (callable)
 * @param A Pointer to the start of input tensor A.
 * @param C Pointer to the start of output scalar.
 * @param A_end Pointer to the end of input tensor A.
 * @param op Unary operation to apply.
 */
template <typename T, class Op>
void scalarRhs(const T *A, T *C, const T *A_end, Op op) {
    for (; A != A_end; A++) {
        op(*A, *C);
    }
}

/**
 * @brief Applies a unary operation element-wise: C[i] = op(A[i])
 * @tparam T Element type
 * @tparam Op Unary operation type (callable)
 * @param A Pointer to the start of input tensor A.
 * @param C Pointer to the start of output tensor C.
 * @param C_end Pointer to the end of output tensor C.
 * @param op Unary operation to apply.
 */
template <typename T, class Op>
void pointwise(const T *A, T *C, const T *C_end, Op op) {
    for (; C != C_end; A++, C++) {
        op(*A, *C);
    }
}

/**
 * @brief Transposes a tensor by copying its contents (no-op for flat arrays).
 * @tparam T Element type
 * @tparam Op Operation (ignored here).
 * @param A Pointer to input tensor.
 * @param C Pointer to output tensor.
 * @param len Number of elements to copy.
 * @param op Operation (placeholder, not applied).
 */
template <typename T, class Op>
void transpose(const T *A, T *C, const T *A_end, Op op) {
    std::copy(A, A_end, C);
}

/**
 * @brief Computes sum of tensor A along a given dimension.
 * @tparam T Element type
 * @param A Pointer to the start of input tensor A.
 * @param C Pointer to the start of output tensor C.
 * @param strideA Stride for A
 * @param strideC Stride for C
 * @param A_offset Offset array per dimension
 * @param N Number of dimensions
 */
template <typename T>
void sum_dim(const T *A, T *C, int *strideA, int *strideC, size_t *A_offset,
             int N) {
    const T *end = A + *A_offset;
    if (N <= 1) {
        for (; A != end; A += *strideA, C += *strideC) {
            *C += *A;
        }
    } else {
        for (; A != end; A += *strideA, C += *strideC) {
            sum_dim(A, C, strideA + 1, strideC + 1, A_offset + 1, N - 1);
        }
    }
}

/**
 * @brief Compile-time recursive version of sum of a tensor along a given
 * dimension gradient.
 * @see sum_dim(const T *A, T *C, int *strideA, int *strideC, size_t *A_offset,
 * int N)
 */
template <typename T, int N>
void sum_dim(const T *A, T *C, int *strideA, int *strideC, size_t *A_offset,
             int _) {
    const T *end = A + *A_offset;
    if constexpr (N <= 1) {
        for (; A != end; A += *strideA, C += *strideC) {
            *C += *A;
        }
    } else {
        for (; A != end; A += *strideA, C += *strideC) {
            sum_dim<T, N - 1>(A, C, strideA + 1, strideC + 1, A_offset + 1, 0);
        }
    }
}

/**
 * @brief Computes the mean of all elements in tensor A.
 * @tparam T Element type
 * @param A Pointer to the start of input tensor A.
 * @param C Pointer to the start of output scalar.
 * @param A_end Pointer to the end of intput tensor A.
 * @param divisor Number of elements
 */
template <typename T> void mean(const T *A, T *C, const T *A_end, T divisor) {
    for (; A != A_end; A++) {
        *C += *A;
    }
    *C /= divisor;
}

/**
 * @brief Computes mean of tensor A along a given dimension.
 * @tparam T Element type
 * @param A Pointer to the start of input tensor A.
 * @param C Pointer to the start of output tensor C.
 * @param strideA Stride for A
 * @param strideC Stride for C
 * @param A_offset Offset array per dimension
 * @param N Number of dimensions
 * @param divisor divisor to compute mean of A (length of dimension summed over)
 * @param C_end Pointer to the end of output tensor C.
 */
template <typename T>
void mean_dim(const T *A, T *C, int *strideA, int *strideC, size_t *A_offset,
              int N, T divisor, T *C_end) {
    sum_dim(A, C, strideA, strideC, A_offset, N);
    for (; C != C_end; C++) {
        *C /= divisor;
    }
}

/**
 * @brief Compile-time recursive version of sum of a tensor along a given
 * dimension gradient.
 * @see mean_dim(const T *A, T *C, int *strideA, int *strideC, size_t *A_offset,
 * int N, T divisor, T *C_end)
 */
template <typename T, int N>
void mean_dim(const T *A, T *C, int *strideA, int *strideC, size_t *A_offset,
              int _, T divisor, T *C_end) {
    sum_dim<T, N>(A, C, strideA, strideC, A_offset, 0);
    for (; C != C_end; C++) {
        *C /= divisor;
    }
}

/**
 * @brief Copies a sliced view of A into C based on offset and stride.
 * @tparam T Element type
 * @param A Pointer to the start of input tensor
 * @param C Pointer to the start of output tensor
 * @param strideA Stride for A
 * @param strideC Stride for C
 * @param start_offset_a Offset to apply to A
 * @param C_offset Size of output slice
 * @param N Number of dimensions
 */
template <typename T>
void slice(const T *A, T *C, int *strideA, int *strideC, size_t *start_offset_a,
           size_t *C_offset, int N) {
    A += *start_offset_a;
    const T *end = C + *C_offset;
    if (N <= 1) {
        for (; C != end; A += *strideA, C += *strideC) {
            *C = *A;
        }
    } else {
        for (; C < end; A += *strideA, C += *strideC) {
            slice(A, C, strideA + 1, strideC + 1, start_offset_a + 1,
                  C_offset + 1, N - 1);
        }
    }
}

/**
 * @brief Compile-time recursive version of slice.
 * @see slice(const T *A, T *C, int *strideA, int *strideC, size_t
 * *start_offset_a, size_t *C_offset, int N)
 */
template <typename T, int N>
void slice(const T *A, T *C, int *strideA, int *strideC, size_t *start_offset_a,
           size_t *C_offset, int _) {
    A += *start_offset_a;
    const T *end = C + *C_offset;
    if constexpr (N <= 1) {
        for (; C != end; A += *strideA, C += *strideC) {
            *C = *A;
        }
    } else {
        for (; C < end; A += *strideA, C += *strideC) {
            slice<T, N - 1>(A, C, strideA + 1, strideC + 1, start_offset_a + 1,
                            C_offset + 1, 0);
        }
    }
}

} // namespace unary
} // namespace Operations
} // namespace kaad
