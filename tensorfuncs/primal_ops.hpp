#pragma once

#include "kernels.hpp" // for Kernels::Null
#include <algorithm>   // for std::copy
#include <cstddef>     // for size_t

namespace kaad {

/**
 * @file tensorfuncs.hpp
 * @brief Tensor operations with NumPy-style broadcasting.
 *
 * Broadcasting rules:
 * - When performing elementwise operations, input tensors are aligned from the
 * trailing dimensions.
 * - For each dimension:
 *   1. If the sizes match, that dimension is compatible.
 *   2. If one tensor has size 1, it is broadcast to match the other.
 *   3. If sizes differ and neither is 1, the shapes are incompatible.
 * - The resulting tensor shape is the elementwise maximum along each dimension.
 *
 * Examples:
 *   (3, 1, 5) and (1, 4, 5) → broadcast to (3, 4, 5)
 *   (2, 3) and (3,) → broadcast to (2, 3)
 *
 * @note These rules are identical to NumPy’s broadcasting semantics.
 */

/**
 * @namespace tensorfuncs::primal
 * @brief Contains primal (e.g. used for forward computation) tensor operations.
 */
namespace tensorfuncs::primal {

/**
 * @namespace kaad::tensorfuncs::adjoint::binary
 */
namespace binary {

template <typename T, class Kernel>
using pointwise_fn = void (*)(const T *A, const T *B, T *C, const T *C_end);

template <typename T, class Kernel>
using flexible_fn = void (*)(const T *A, const T *B, T *C, int *strideA,
                             int *strideB, int *strideC, size_t *c_dim_offset,
                             int c_rank);

template <typename T>
using dot_fn = void (*)(const T *A, const T *B, T *C, const T *A_end);

template <typename T>
using matmul_fn = void (*)(const T *A, const T *B, T *C, int a_rows, int b_cols,
                           int shared_dim, int *strideA, int *strideB,
                           int *strideC);

template <typename T>
using batch_matmul_fn = void (*)(const T *A, const T *B, T *C, int *strideA,
                                 int *strideB, int *strideC, int *c_shape,
                                 int a_dim_offset, int b_dim_offset,
                                 int shared_dim, int c_rank);

/**
 * @brief Applies Op(A,B) to A(tensor) and B(scalar).
 * @pre @p A and @p C have the same shape and @p B is scalar.
 * @tparam T Element type
 * @tparam Kernel A struct containing a static binary function ('Op').
 * @param[in] A Pointer to the start of A(tensor).
 * @param[in] B Pointer to B(scalar).
 * @param[out] C Pointer to the start of C(tensor).
 * @param C_end Pointer to the end of @p C.
 */
template <typename T, class Kernel>
void scalarRhs(const T *A, const T *B, T *C, const T *C_end) {
    for (; C != C_end; A++, C++) {
        Kernel::Op(*A, *B, *C);
    }
}

/**
 * @brief Applies Op(A,B) to A(scalar) and B(tensor).
 * @pre @p B and @p C have the same shape and @p A is scalar.
 * @tparam T Element type
 * @tparam Kernel A struct containing a static binary function ('Op').
 * @param[in] A Pointer to A(scalar).
 * @param[in] B Pointer to the start of B(tensor).
 * @param[out] C Pointer to the start of C(tensor).
 * @param C_end Pointer to the end of @p C.
 */
template <typename T, class Kernel>
void scalarLhs(const T *A, const T *B, T *C, const T *C_end) {
    for (; C != C_end; B++, C++) {
        Kernel::Op(*A, *B, *C);
    }
}

/**
 * @brief Applies Op(A,B) to A(tensor) and B(tensor).
 * @pre @p A, @p B and @p C have the same shape.
 * @tparam T Element type
 * @tparam Kernel A struct containing a static binary function ('Op').
 * @param[in] A Pointer to the start of A(tensor).
 * @param[in] B Pointer to the start of B(tensor).
 * @param[out] C Pointer to the start of C(tensor).
 * @param C_end Pointer to the end of @p C.
 */
template <typename T, class Kernel>
void pointwise(const T *A, const T *B, T *C, const T *C_end) {
    for (; C != C_end; A++, B++, C++) {
        Kernel::Op(*A, *B, *C);
    }
}

/**
 * @brief Applies Op(A,B) to A(tensor) and B(tensor).
 * @pre @p C shape is the result of broadcasting @p A and @p B.
 * @tparam T Element type
 * @tparam Kernel A struct containing a static binary function ('Op').
 * @param[in] A Pointer to the start of A(tensor).
 * @param[in] B Pointer to the start of B(tensor).
 * @param[out] C Pointer to the start of C(tensor).
 * @param strideA Stride array of A.
 * @param strideB Stride array of B.
 * @param strideC Stride array of C.
 * @param c_dim_offset Offset to the end of @p C per dimension.
 * @param c_rank Number of dimensions of C.
 */
template <typename T, class Kernel>
void flexible(const T *A, const T *B, T *C, int *strideA, int *strideB,
              int *strideC, size_t *c_dim_offset, int rank) {
    const T *end = C + *c_dim_offset;
    if (rank <= 1) {
        for (; C != end; A += *strideA, B += *strideB, C += *strideC) {
            Kernel::Op(*A, *B, *C);
        }
    } else {
        for (; C < end; A += *strideA, B += *strideB, C += *strideC) {
            flexible<T, Kernel>(A, B, C, strideA + 1, strideB + 1, strideC + 1,
                                c_dim_offset + 1, rank - 1);
        }
    }
}

/**
 * @brief Compile-time recursive version of flexible.
 * @see void flexible(const T *A, const T *B, T *C, int *strideA, int
 * *strideB, int *strideC, size_t *c_dim_offset, int rank)
 */
template <typename T, class Kernel, int rank>
void flexible(const T *A, const T *B, T *C, int *strideA, int *strideB,
              int *strideC, size_t *c_dim_offset, int _) {
    const T *end = C + *c_dim_offset;
    if constexpr (rank <= 1) {
        for (; C != end; A += *strideA, B += *strideB, C += *strideC) {
            Kernel::Op(*A, *B, *C);
        }
    } else {
        for (; C < end; A += *strideA, B += *strideB, C += *strideC) {
            flexible<T, Kernel, rank - 1>(A, B, C, strideA + 1, strideB + 1,
                                          strideC + 1, c_dim_offset + 1, 0);
        }
    }
}

/**
 * @brief Computes the dot product of A and B into C.
 * @pre @p A is a vector and B and C are scalars.
 * @tparam T Element type
 * @tparam Kernel (Only needed for signature).
 * @param[in] A Pointer to the start of A(vector).
 * @param[in] B Pointer to B(scalar)
 * @param[out] C Pointer to C(scalar).
 * @param A_end Pointer to the end of @p A.
 */
template <typename T>
void scalarDot(const T *A, const T *B, T *C, const T *A_end) {
    for (; A != A_end; A++) {
        *C += *A * (*B);
    }
}

/**
 * @brief Computes the dot product of A and B into C.
 * @pre @p A and @p B are vectors and C is a scalar.
 * @tparam T Element type
 * @tparam Kernel (Only needed for signature).
 * @param[in] A Pointer to the start of A(vector).
 * @param[in] B Pointer to the start of B(vector)
 * @param[out] C Pointer to C(scalar).
 * @param A_end Pointer to the end of @p A.
 */
template <typename T> void dot(const T *A, const T *B, T *C, const T *A_end) {
    for (; A != A_end; A++, B++) {
        *C += *A * (*B);
    }
}

/**
 * @brief Computes matrix product of A and B into C.
 * @pre @p A, @p B and @p C have compatible shapes.
 * @tparam T Element type
 * @param[in] A Pointer to the start of A(matrix).
 * @param[in] B Pointer to the start of B(matrix).
 * @param[out] C Pointer to the start of C(matrix).
 * @param a_rows Number of rows in @p A.
 * @param b_cols Number of columns in @p B.
 * @param strideA Stride array of A.
 * @param strideB Stride array of B.
 * @param strideC Stride array of C.
 */
template <typename T>
void matmul(const T *A, const T *B, T *C, int a_rows, int b_cols,
            int shared_dim, int *strideA, int *strideB, int *strideC) {
    const T *rowA;
    const T *colB;
    const T *elemB;
    for (int a_idx = 0; a_idx < a_rows;
         a_idx++, A += strideA[0], C += strideC[0]) {
        colB = B;
        for (int b_idx = 0; b_idx < b_cols;
             b_idx++, colB += strideB[1], C += strideC[1]) {
            rowA = A;
            elemB = colB;
            for (int i = 0; i < shared_dim;
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
 * @param a_dim_offset Step size for A.
 * @param b_dim_offset Step size for B.
 * @param shared_dim Shared inner dimension.
 * @param c_rank Number of dimension of C.
 */
template <typename T>
void batch_matmul(const T *A, const T *B, T *C, int *strideA, int *strideB,
                  int *strideC, int *c_shape, int a_dim_offset,
                  int b_dim_offset, int shared_dim, int c_rank) {
    const T *end = C + (*c_shape) * (*strideC);
    if (c_rank <= 1) {
        for (int i = 0; i < *c_shape;
             i++, A += *strideA, B += *strideB, C += *strideC) {
            const T *rowA = A, *colB = B;
            for (int j = 0; j < shared_dim;
                 j++, rowA += a_dim_offset, colB += b_dim_offset) {
                *C += (*rowA) * (*colB);
            }
        }
    } else {
        for (int i = 0; i < *c_shape;
             i++, A += *strideA, B += *strideB, C += *strideC) {
            batch_matmul(A, B, C, strideA + 1, strideB + 1, strideC + 1,
                         c_shape + 1, a_dim_offset, b_dim_offset, shared_dim,
                         c_rank - 1);
        }
    }
}

/**
 * @brief Compile-time recursive version of batched matrix multiplication.
 * @see void batch_matmul(const T *A, const T *B, T *C, int *strideA, int
 * *strideB, int *strideC, int *c_shape, int a_dim_offset, int b_dim_offset,
 * int shared_dim, int c_rank)
 */
template <typename T, int c_rank>
void batch_matmul(const T *A, const T *B, T *C, int *strideA, int *strideB,
                  int *strideC, int *c_shape, int a_dim_offset,
                  int b_dim_offset, int shared_dim, int _) {
    const T *end = C + (*c_shape) * (*strideC);
    if constexpr (c_rank <= 1) {
        for (int i = 0; i < *c_shape;
             i++, A += *strideA, B += *strideB, C += *strideC) {
            const T *rowA = A, *colB = B;
            for (int j = 0; j < shared_dim;
                 j++, rowA += a_dim_offset, colB += b_dim_offset) {
                *C += (*rowA) * (*colB);
            }
        }
    } else {
        for (int i = 0; i < *c_shape;
             i++, A += *strideA, B += *strideB, C += *strideC) {
            batch_matmul<T, c_rank - 1>(A, B, C, strideA + 1, strideB + 1,
                                        strideC + 1, c_shape + 1, a_dim_offset,
                                        b_dim_offset, shared_dim, 0);
        }
    }
}

} // namespace binary

/**
 * @namespace kaad::Operations::unary
 */
namespace unary {

template <typename T, class Kernel>
using pointwise_fn = void (*)(const T *A, T *C, const T *C_end);

template <typename T>
using sum_dim_fn = void (*)(const T *A, T *C, int *strideA, int *strideC,
                            size_t *A_offset, int c_rank);

template <typename T>
using mean_fn = void (*)(const T *A, T *C, const T *A_end, T divisor);

template <typename T>
using mean_dim_fn = void (*)(const T *A, T *C, int *strideA, int *strideC,
                             size_t *A_offset, int c_rank, T divisor,
                             const T *C_end);

template <typename T>
using slice_fn = void (*)(const T *A, T *C, int *strideA, int *strideC,
                          size_t *start_offset_a, size_t *c_dim_offset,
                          int c_rank);

/**
 * @brief Performs no operation (copies A to C).
 * @tparam T Element type
 * @tparam Op Operation (ignored here).
 * @param A Pointer to input tensor.
 * @param C Pointer to output tensor.
 * @param A_end Pointer to the end of A.
 * @param op Instance of the callable class (ignored here).
 */
template <typename T, class Kernel = Kernels::Null>
void noop(const T *A, T *C, const T *A_end) {
    std::copy(A, A_end, C);
}

/**
 * @brief Applies a unary operation to A(tensor).
 * @tparam T Element type
 * @tparam Kernel A struct containing a static unary function ('Op').
 * @param[in] A Pointer to the start of A(tensor).
 * @param[out] C Pointer to C(scalar).
 * @param A_end Pointer to the end of A.
 */
template <typename T, class Kernel>
void scalarOut(const T *A, T *C, const T *A_end) {
    for (; A != A_end; A++) {
        Kernel::Op(*A, *C);
    }
}

/**
 * @brief Applies a unary operation to A(tensor).
 * @tparam T Element type
 * @tparam Kernel A struct containing a static unary function ('Op').
 * @param[in] A Pointer to the start of A(tensor).
 * @param[out] C Pointer to the start of C(tensor)
 * @param A_end Pointer to the end of C.
 * @param op Instance of the callable class.
 */
template <typename T, class Kernel>
void pointwise(const T *A, T *C, const T *C_end) {
    for (; C != C_end; A++, C++) {
        Kernel::Op(*A, *C);
    }
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
 * @param a_rank Number of dimensions of A.
 */
template <typename T>
void sum_dim(const T *A, T *C, int *strideA, int *strideC, size_t *A_offset,
             int a_rank) {
    const T *end = A + *A_offset;
    if (a_rank <= 1) {
        for (; A != end; A += *strideA, C += *strideC) {
            *C += *A;
        }
    } else {
        for (; A != end; A += *strideA, C += *strideC) {
            sum_dim(A, C, strideA + 1, strideC + 1, A_offset + 1, a_rank - 1);
        }
    }
}

/**
 * @brief Compile-time recursive version of sum_dim.
 * @see void sum_dim(const T *A, T *C, int *strideA, int *strideC, size_t
 * *A_offset, int a_rank)
 */
template <typename T, int a_rank>
void sum_dim(const T *A, T *C, int *strideA, int *strideC, size_t *A_offset,
             int _) {
    const T *end = A + *A_offset;
    if constexpr (a_rank <= 1) {
        for (; A != end; A += *strideA, C += *strideC) {
            *C += *A;
        }
    } else {
        for (; A != end; A += *strideA, C += *strideC) {
            sum_dim<T, a_rank - 1>(A, C, strideA + 1, strideC + 1, A_offset + 1,
                                   0);
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
 * @param a_rank Number of dimensions in A.
 * @param divisor divisor to compute mean of A (length of dimension summed over)
 * @param C_end Pointer to the end of output tensor C.
 */
template <typename T>
void mean_dim(const T *A, T *C, int *strideA, int *strideC, size_t *A_offset,
              int a_rank, T divisor, const T *C_end) {
    sum_dim(A, C, strideA, strideC, A_offset, a_rank);
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
              int _, T divisor, const T *C_end) {
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
 * @param c_dim_offset Size of output slice.
 * @param rank Number of dimensions in A and C.
 */
template <typename T>
void slice(const T *A, T *C, int *strideA, int *strideC, size_t *start_offset_a,
           size_t *c_dim_offset, int rank) {
    A += *start_offset_a;
    const T *end = C + *c_dim_offset;
    if (rank <= 1) {
        for (; C != end; A += *strideA, C += *strideC) {
            *C = *A;
        }
    } else {
        for (; C < end; A += *strideA, C += *strideC) {
            slice(A, C, strideA + 1, strideC + 1, start_offset_a + 1,
                  c_dim_offset + 1, rank - 1);
        }
    }
}

/**
 * @brief Compile-time recursive version of slice.
 * @see slice(const T *A, T *C, int *strideA, int *strideC, size_t
 * *start_offset_a, size_t *c_dim_offset, int rank)
 */
template <typename T, int rank>
void slice(const T *A, T *C, int *strideA, int *strideC, size_t *start_offset_a,
           size_t *c_dim_offset, int _) {
    A += *start_offset_a;
    const T *end = C + *c_dim_offset;
    if constexpr (rank <= 1) {
        for (; C != end; A += *strideA, C += *strideC) {
            *C = *A;
        }
    } else {
        for (; C < end; A += *strideA, C += *strideC) {
            slice<T, rank - 1>(A, C, strideA + 1, strideC + 1,
                               start_offset_a + 1, c_dim_offset + 1, 0);
        }
    }
}

} // namespace unary
} // namespace tensorfuncs::primal
} // namespace kaad
