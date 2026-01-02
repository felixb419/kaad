#pragma once

#include <algorithm> // for std::copy
#include <cstddef>   // for size_t

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
 * @namespace tensorfuncs::primal
 * @brief Contains primal (e.g. used for forward computation) tensor operations.
 */
namespace tensorfuncs::primal {

/**
 * @namespace kaad::tensorfuncs::adjoint::binary
 */
namespace binary {

/**
 * @brief Applies Op(A,B) to A(tensor) and B(scalar).
 * @pre @p A and @p C have the same shape and @p B is scalar.
 * @tparam T Element type
 * @tparam Op A Callable binary operation.
 * @param[in] A Pointer to the start of A(tensor).
 * @param[in] B Pointer to B(scalar).
 * @param[out] C Pointer to the start of C(tensor).
 * @param C_end Pointer to the end of @p C.
 * @param op Instance of the callable class.
 */
template <typename T, class Op>
void scalarRhs(const T *A, const T *B, T *C, const T *C_end, Op op) {
    for (; C != C_end; A++, C++) {
        op(*A, *B, *C);
    }
}

/**
 * @brief Applies Op(A,B) to A(scalar) and B(tensor).
 * @pre @p B and @p C have the same shape and @p A is scalar.
 * @tparam T Element type
 * @tparam Op A Callable binary operation.
 * @param[in] A Pointer to A(scalar).
 * @param[in] B Pointer to the start of B(tensor).
 * @param[out] C Pointer to the start of C(tensor).
 * @param C_end Pointer to the end of @p C.
 * @param op Instance of the callable class.
 */
template <typename T, class Op>
void scalarLhs(const T *A, const T *B, T *C, const T *C_end, Op op) {
    for (; C != C_end; B++, C++) {
        op(*A, *B, *C);
    }
}

/**
 * @brief Applies Op(A,B) to A(tensor) and B(tensor).
 * @pre @p A, @p B and @p C have the same shape.
 * @tparam T Element type
 * @tparam Op A Callable binary operation.
 * @param[in] A Pointer to the start of A(tensor).
 * @param[in] B Pointer to the start of B(tensor).
 * @param[out] C Pointer to the start of C(tensor).
 * @param C_end Pointer to the end of @p C.
 * @param op Instance of the callable class.
 */
template <typename T, class Op>
void pointwise(const T *A, const T *B, T *C, const T *C_end, Op op) {
    for (; C != C_end; A++, B++, C++) {
        op(*A, *B, *C);
    }
}

/**
 * @brief Applies Op(A,B) to A(tensor) and B(tensor).
 * @pre @p C shape is the result of broadcasting @p A and @p B.
 * @tparam T Element type
 * @tparam Op A Callable binary operation.
 * @param[in] A Pointer to the start of A(tensor).
 * @param[in] B Pointer to the start of B(tensor).
 * @param[out] C Pointer to the start of C(tensor).
 * @param strideA Stride array of A.
 * @param strideB Stride array of B.
 * @param strideC Stride array of C.
 * @param C_offset Offset to the end of @p C per dimension.
 * @param N Number of dimensions of C.
 * @param op Instance of the callable class.
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
 * @brief Compile-time recursive version of flexible.
 * @see void flexible(const T *A, const T *B, T *C, int *strideA, int *strideB,
 * int *strideC, size_t *C_offset, int N, Op op)
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
 * @brief Computes the dot product of A and B into C.
 * @pre @p A is 1-dimensional and B and C is a scalar.
 * @tparam T Element type
 * @tparam Op (Only needed for signature).
 * @param[in] A Pointer to the start of A(vector).
 * @param[in] B Pointer to B(scalar)
 * @param[out] C Pointer to C(scalar).
 * @param A_end Pointer to the end of @p A.
 * @param _ (Only needed for signature).
 */
template <typename T, class Op>
void scalarDot(const T *A, const T *B, T *C, const T *A_end, Op _) {
    for (; A != A_end; A++) {
        *C += *A * (*B);
    }
}

/**
 * @brief Computes the dot product of A and B into C.
 * @pre @p A and @p B are 1-dimensional and C is a scalar.
 * @tparam T Element type
 * @tparam Op (Only needed for signature).
 * @param[in] A Pointer to the start of A(vector).
 * @param[in] B Pointer to the start of B(vector)
 * @param[out] C Pointer to C(scalar).
 * @param A_end Pointer to the end of @p A.
 * @param _ (Only needed for signature).
 */
template <typename T, class Op>
void dot(const T *A, const T *B, T *C, const T *A_end, Op _) {
    for (; A != A_end; A++, B++) {
        *C += *A * (*B);
    }
}

/**
 * @brief Computes Matrix product of A and B into C.
 * @pre @p A, @p B and @p C have compatible shapes.
 * @tparam T Element type
 * @param[in] A Pointer to the start of A(matrix).
 * @param[in] B Pointer to the start of B(matrix).
 * @param[out] C Pointer to the start of C(matrix).
 * @param a_dim Number of rows in @p A.
 * @param b_dim Number of columns in @p B.
 * @param strideA Stride array of A.
 * @param strideB Stride array of B.
 * @param strideC Stride array of C.
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
 * @brief Computes Matrix product of A and B into C (treats additional
 * dimensions as batch dimensions).
 * @pre @p A, @p B and @p C have compatible shapes.
 * @tparam T Element type
 * @param[in] A Pointer to the start of A(tensor).
 * @param[in] B Pointer to the start of B(tensor).
 * @param[out] C Pointer to the start of C(tensor).
 * @param strideA Stride array of A.
 * @param strideB Stride array of B.
 * @param strideC Stride array of C.
 * @param Pointer to shape array of C.
 * @param a_off Step size for A.
 * @param b_off Step size for B.
 * @param k Shared inner dimension.
 * @param N Number of dimension of C.
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
 * @see void batch_matmul(const T *A, const T *B, T *C, int *strideA, int
 * *strideB, int *strideC, int *c_shape, int a_off, int b_off, int k, int N)
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
 */
namespace unary {

/**
 * @brief Applies a unary operation to A(tensor).
 * @tparam T Element type
 * @tparam Op A Callable unary operation.
 * @param[in] A Pointer to the start of A(tensor).
 * @param[out] C Pointer to C(scalar).
 * @param A_end Pointer to the end of A.
 * @param op Instance of the callable class.
 */
template <typename T, class Op>
void scalarOut(const T *A, T *C, const T *A_end, Op op) {
    for (; A != A_end; A++) {
        op(*A, *C);
    }
}

/**
 * @brief Applies a unary operation to A(tensor).
 * @tparam T Element type
 * @tparam Op A Callable unary operation.
 * @param[in] A Pointer to the start of A(tensor).
 * @param[out] C Pointer to the start of C(tensor)
 * @param A_end Pointer to the end of C.
 * @param op Instance of the callable class.
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
 * @brief Sums A(Tensor) along a dimension into C(Tensor).
 * @pre Shape of C needs to be same as A with relevant dimension removed.
 * @tparam T Element type
 * @param A Pointer to the start of A(Tensor).
 * @param C Pointer to the start of C(Tensor).
 * @param strideA Stride array of A.
 * @param strideB Stride array of B.
 * @param strideC Stride array of C.
 * @param A_offset Offset to the end of @p A per dimension.
 * @param N Number of dimensions of A.
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
 * @brief Compile-time recursive version of sum_dim.
 * @see void sum_dim(const T *A, T *C, int *strideA, int *strideC, size_t
 * *A_offset, int N)
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
 * @brief Computes the mean of all elements in A(tensor).
 * @tparam T Element type
 * @param[in] A Pointer to the start of A(tensor).
 * @param[out] A Pointer to the start of C(tensor).
 * @param A_end Pointer to the end A.
 * @param divisor Number of elements
 */
template <typename T> void mean(const T *A, T *C, const T *A_end, T divisor) {
    for (; A != A_end; A++) {
        *C += *A;
    }
    *C /= divisor;
}

/**
 * @brief Computes mean of A(tensor) along a given dimension.
 * @tparam T Element type
 * @param[in] A Pointer to the start of A(tensor).
 * @param[out] C Pointer to the start of C(tensor).
 * @param strideA Stride array for A.
 * @param strideC Stride array for C.
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
 * @brief Compile-time recursive version of mean_dim.
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
 * @param[in] A Pointer to the start of A(tensor).
 * @param[out] C Pointer to the start of C(tensor).
 * @param strideA Stride array for A.
 * @param strideC Stride array for C.
 * @param start_offset_a Offset to apply to A.
 * @param C_offset Size of output slice.
 * @param N Number of dimensions.
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
} // namespace tensorfuncs::primal
} // namespace kaad
