#pragma once

#include "primal_ops.hpp" // for batch_matmul, matmul
#include <cstddef>        // for size_t

namespace kaad {

/**
 * @namespace tensorfuncs::adjoint
 * @brief Contains adjoint (e.g. used for backward computation) tensor
 * operations.
 */
namespace tensorfuncs::adjoint {

/**
 * @namespace kaad::tensoruncs::adjoint
 */
namespace binary {

template <class Kernel>
using pointwise_fn = void (*)(const typename Kernel::value_type *A,
                              typename Kernel::value_type *dA,
                              const typename Kernel::value_type *B,
                              typename Kernel::value_type *dB,
                              const typename Kernel::value_type *C,
                              const typename Kernel::value_type *dC,
                              const typename Kernel::value_type *end);

template <class Kernel>
using flexible_fn = void (*)(
    const typename Kernel::value_type *A, typename Kernel::value_type *dA,
    const typename Kernel::value_type *B, typename Kernel::value_type *dB,
    const typename Kernel::value_type *C, const typename Kernel::value_type *dC,
    int *strideA, int *strideB, int *strideC, size_t *c_dim_offset, int c_rank);

template <typename T>
using dot_fn = void (*)(const T *A, T *dA, const T *B, T *dB, const T *C,
                        const T *dC, const T *A_end);

template <typename T>
using matmul_fn = void (*)(const T *A, T *dA, const T *B, T *dB, const T *C,
                           const T *dC, int *a_rows, int *b_cols,
                           int *shared_dim, int *strideA, int *strideB,
                           int *strideC);

template <typename T>
using batch_matmul_fn = void (*)(const T *A, T *dA, const T *B, T *dB,
                                 const T *C, const T *dC, int **strideA,
                                 int **strideB, int **strideC, int **c_shape,
                                 int *a_dim_offset, int *b_dim_offset,
                                 int *shared_dim, int c_rank);

/**
 * @brief Accumulates the gradient of Op(A,B), A(tensor), B(scalar).
 * @pre @p A and @p C have the same shape and @p B is scalar.
 * @pre Every operand must have the same shape as their gradient.
 * @tparam Kernel A struct containing a static binary funcion ('Grad').
 * @param[in] A Pointer to the start of A(tensor).
 * @param[out] dA Pointer to the start of the gradient w.r.t. @p A.
 * @param[in] B Pointer to the start of B(scalar).
 * @param[out] dB Pointer to the start of the gradient w.r.t. @p B.
 * @param[in] C Pointer to the start of C(tensor).
 * @param[in] dC Pointer to the start of the gradient w.r.t. @p C.
 * @param C_end Pointer to the end of @p C.
 */
template <class Kernel>
void scalarRhs(const typename Kernel::value_type *A,
               typename Kernel::value_type *dA,
               const typename Kernel::value_type *B,
               typename Kernel::value_type *dB,
               const typename Kernel::value_type *C,
               const typename Kernel::value_type *dC,
               const typename Kernel::value_type *C_end) {
    for (; C != C_end; A++, dA++, C++, dC++) {
        Kernel::Grad(*A, *dA, *B, *dB, *C, *dC);
    }
}

/**
 * @brief Accumulates the gradient of Op(A,B), A(scalar), B(tensor).
 * @pre @p B and @p C have the same shape and @p A is scalar.
 * @pre Every operand must have the same shape as their gradient.
 * @tparam Kernel A struct containing a static binary funcion ('Grad').
 * @param[in] A Pointer to the start of A(scalar).
 * @param[out] dA Pointer to the start of the gradient w.r.t. @p A.
 * @param[in] B Pointer to the start of B(tensor).
 * @param[out] dB Pointer to the start of the gradient w.r.t. @p B.
 * @param[in] C Pointer to the start of C(tensor).
 * @param[in] dC Pointer to the start of the gradient w.r.t. @p C.
 * @param C_end Pointer to the end of @p C.
 */
template <class Kernel>
void scalarLhs(const typename Kernel::value_type *A,
               typename Kernel::value_type *dA,
               const typename Kernel::value_type *B,
               typename Kernel::value_type *dB,
               const typename Kernel::value_type *C,
               const typename Kernel::value_type *dC,
               const typename Kernel::value_type *C_end) {
    for (; C != C_end; B++, dB++, C++, dC++) {
        Kernel::Grad(*A, *dA, *B, *dB, *C, *dC);
    }
}

/**
 * @brief Accumulates the gradient of Op(A,B), A(tensor), B(tensor).
 * @pre @p A, @p B and @p C have the same shape.
 * @pre Every operand must have the same shape as their gradient.
 * @tparam Kernel A struct containing a static binary funcion ('Grad').
 * @param[in] A Pointer to the start of A(tensor).
 * @param[out] dA Pointer to the start of the gradient w.r.t. @p A.
 * @param[in] B Pointer to the start of B(tensor).
 * @param[out] dB Pointer to the start of the gradient w.r.t. @p B.
 * @param[in] C Pointer to the start of C(tensor).
 * @param[in] dC Pointer to the start of the gradient w.r.t. @p C.
 * @param C_end Pointer to the end of @p C.
 */
template <class Kernel>
void pointwise(const typename Kernel::value_type *A,
               typename Kernel::value_type *dA,
               const typename Kernel::value_type *B,
               typename Kernel::value_type *dB,
               const typename Kernel::value_type *C,
               const typename Kernel::value_type *dC,
               const typename Kernel::value_type *C_end) {
    for (; C != C_end; A++, dA++, B++, dB++, C++, dC++) {
        Kernel::Grad(*A, *dA, *B, *dB, *C, *dC);
    }
}

/**
 * @brief Accumulates the gradient of Op(A,B), A(tensor), B(tensor).
 * @pre @p C shape is the result of broadcasting @p A and @p B.
 * @pre Every operand must have the same shape as their gradient.
 * @tparam Kernel A struct containing a static binary funcion ('Grad').
 * @param[in] A Pointer to the start of A(tensor).
 * @param[out] dA Pointer to the start of the gradient w.r.t. @p A.
 * @param[in] B Pointer to the start of B(tensor).
 * @param[out] dB Pointer to the start of the gradient w.r.t. @p B.
 * @param[in] C Pointer to the start of C(tensor).
 * @param[in] dC Pointer to the start of the gradient w.r.t. @p C.
 * @param strideA Stride array of A.
 * @param strideB Stride array of B.
 * @param strideC Stride array of C.
 * @param c_dim_offset Offset to the end of @p C per dimension.
 * @param c_rank Number of dimensions of C.
 */
template <class Kernel>
void flexible(const typename Kernel::value_type *A,
              typename Kernel::value_type *dA,
              const typename Kernel::value_type *B,
              typename Kernel::value_type *dB,
              const typename Kernel::value_type *C,
              const typename Kernel::value_type *dC, int *strideA, int *strideB,
              int *strideC, size_t *c_dim_offset, int c_rank) {
    const typename Kernel::value_type *end = C + *c_dim_offset;
    if (c_rank <= 1) {
        for (; C != end; A += *strideA, B += *strideB, C += *strideC,
                         dA += *strideA, dB += *strideB, dC += *strideC) {
            Kernel::Grad(*A, *dA, *B, *dB, *C, *dC);
        }
    } else {
        for (; C < end; A += *strideA, B += *strideB, C += *strideC,
                        dA += *strideA, dB += *strideB, dC += *strideC) {
            flexible<Kernel>(A, dA, B, dB, C, dC, strideA + 1, strideB + 1,
                             strideC + 1, c_dim_offset + 1, c_rank - 1);
        }
    }
}

/**
 * @brief Compile-time recursive version of runtime @ref flexible().
 */
template <class Kernel, int c_rank>
void flexible(const typename Kernel::value_type *A,
              typename Kernel::value_type *dA,
              const typename Kernel::value_type *B,
              typename Kernel::value_type *dB,
              const typename Kernel::value_type *C,
              const typename Kernel::value_type *dC, int *strideA, int *strideB,
              int *strideC, size_t *c_dim_offset, int _) {
    const typename Kernel::value_type *end = C + *c_dim_offset;
    if constexpr (c_rank <= 1) {
        for (; C != end; A += *strideA, B += *strideB, C += *strideC,
                         dA += *strideA, dB += *strideB, dC += *strideC) {
            Kernel::Grad(*A, *dA, *B, *dB, *C, *dC);
        }
    } else {
        for (; C != end; A += *strideA, B += *strideB, C += *strideC,
                         dA += *strideA, dB += *strideB, dC += *strideC) {
            flexible<Kernel, c_rank - 1>(A, dA, B, dB, C, dC, strideA + 1,
                                         strideB + 1, strideC + 1,
                                         c_dim_offset + 1, 0);
        }
    }
}

/**
 * @brief Accumulates the gradient of the dot-product of A(vector) and
 * B(scalar).
 * @pre @p A is a vector and B and C are scalars.
 * @pre Every operand must have the same shape as their gradient.
 * @tparam T Element type.
 * @tparam Kernel (Only neede for signature)
 * @param[in] A Pointer to the start of A(vector).
 * @param[out] dA Pointer to the start of the gradient w.r.t. @p A.
 * @param[in] B Pointer to the start of B(scalar).
 * @param[out] dB Pointer to the start of the gradient w.r.t. @p B.
 * @param[in] C Pointer to the start of C(scalar).
 * @param[in] dC Pointer to the start of the gradient w.r.t. @p C.
 * @param A_end Pointer to the end of @p A.
 */
template <typename T>
void scalarDot(const T *A, T *dA, const T *B, T *dB, const T *C, const T *dC,
               const T *A_end) {
    for (; A != A_end; A++, dA++) {
        *dA += *dC * (*B);
        *dB += *dC * (*A);
    }
}

/**
 * @brief Accumulates the gradient of the dot-product of A(vector) and
 * B(vector).
 * @pre @p A and B are vectors and C is a scalar.
 * @pre Every operand must have the same shape as their gradient.
 * @tparam T Element type.
 * @tparam Kernel (Only neede for signature)
 * @param[in] A Pointer to the start of A(vector).
 * @param[out] dA Pointer to the start of the gradient w.r.t. @p A.
 * @param[in] B Pointer to the start of B(vector).
 * @param[out] dB Pointer to the start of the gradient w.r.t. @p B.
 * @param[in] C Pointer to the start of C(scalar).
 * @param[in] dC Pointer to the start of the gradient w.r.t. @p C.
 * @param A_end Pointer to the end of @p A.
 */
template <typename T>
void dot(const T *A, T *dA, const T *B, T *dB, const T *C, const T *dC,
         const T *A_end) {
    for (; A != A_end; A++, dA++, B++, dB++) {
        *dA += *dC * (*B);
        *dB += *dC * (*A);
    }
}

/**
 * @brief Accumulates the gradient of Op(A,B), A(matrix), B(matrix).
 * @pre @p A, @p B and @p C have compatible shapes.
 * @pre Every operand must have the same shape as their gradient.
 * @tparam T Element type.
 * @param[in] A Pointer to the start of A(matrix).
 * @param[out] dA Pointer to the start of the gradient w.r.t. @p A.
 * @param[in] B Pointer to the start of B(matrix).
 * @param[out] dB Pointer to the start of the gradient w.r.t. @p B.
 * @param[in] C Pointer to the start of C(matrix).
 * @param[in] dC Pointer to the start of the gradient w.r.t. @p C.
 * @param a_rows Number of rows in @p A.
 * @param b_cols Number of columns in @p B.
 * @param shared_dim Length of the shared dimension of @p A and @p B.
 * @param strideA Stride array of A.
 * @param strideB Stride array of B.
 * @param strideC Stride array of C.
 */
template <typename T>
void matmul(const T *A, T *dA, const T *B, T *dB, const T *C, const T *dC,
            int *a_rows, int *b_cols, int *shared_dim, int *strideA,
            int *strideB, int *strideC) {
    // dA = dC * B^T
    tensorfuncs::primal::binary::matmul(dC, B, dA, a_rows[0], b_cols[0],
                                        shared_dim[0], strideC, strideB,
                                        strideA);
    // dB = A^T * dC
    tensorfuncs::primal::binary::matmul(A, dC, dB, a_rows[1], b_cols[1],
                                        shared_dim[1], strideA + 2, strideC + 2,
                                        strideB + 2);
}

/**
 * @brief Accumulates the gradient of Op(A,B), A(tensor), B(tensor).
 * @pre @p A, @p B and @p C have compatible shapes.
 * @pre Every operand must have the same shape as their gradient.
 * @tparam T Element type.
 * @param[in] A Pointer to the start of A(tensor).
 * @param[out] dA Pointer to the start of the gradient w.r.t. @p A.
 * @param[in] B Pointer to the start of B(tensor).
 * @param[out] dB Pointer to the start of the gradient w.r.t. @p B.
 * @param[in] C Pointer to the start of C(tensor).
 * @param[in] dC Pointer to the start of the gradient w.r.t. @p C.
 * @param strideA Pointer to stride arrays ({C.stride, A.stride^T}).
 * @param strideB Pointer to stride arrays ({B.stride^T, C.stride}).
 * @param strideC Pointer to stride arrays ({A.stride, B.stride}).
 * @param c_shape Pointer to shape arrays ({A.shape, B.shape}).
 * @param a_dim_offset Pointer to array of step sizes ({C, A^T}).
 * @param b_dim_offset Pointer to array of step sizes ({B^T, C}).
 * @param shared_dim Pointer to array of length of shared dimensions.
 * @param c_rank Number of dimensions of @p C.
 */
template <typename T>
void batch_matmul(const T *A, T *dA, const T *B, T *dB, const T *C, const T *dC,
                  int **strideA, int **strideB, int **strideC, int **c_shape,
                  int *a_dim_offset, int *b_dim_offset, int *shared_dim,
                  int c_rank) {
    // dA = dC * B^T
    tensorfuncs::primal::binary::batch_matmul<T>(
        dC, B, dA, strideC[0], strideB[0], strideA[0], c_shape[0],
        a_dim_offset[0], b_dim_offset[0], shared_dim[0], c_rank);
    // dB = A^T * dC
    tensorfuncs::primal::binary::batch_matmul<T>(
        A, dC, dB, strideA[1], strideC[1], strideB[1], c_shape[1],
        a_dim_offset[1], b_dim_offset[1], shared_dim[1], c_rank);
}

/**
 * @brief Compile-time recursive version of runtime @ref batch_matmul().
 */
template <typename T, int c_rank>
void batch_matmul(const T *A, T *dA, const T *B, T *dB, const T *C, const T *dC,
                  int **strideA, int **strideB, int **strideC, int **c_shape,
                  int *a_dim_offset, int *b_dim_offset, int *shared_dim,
                  int _) {
    // dA = dC * B^T
    tensorfuncs::primal::binary::batch_matmul<T, c_rank>(
        dC, B, dA, strideC[0], strideB[0], strideA[0], c_shape[0],
        a_dim_offset[0], b_dim_offset[0], shared_dim[0], 0);
    // dB = A^T * dC
    tensorfuncs::primal::binary::batch_matmul<T, c_rank>(
        A, dC, dB, strideA[1], strideC[1], strideB[1], c_shape[1],
        a_dim_offset[1], b_dim_offset[1], shared_dim[1], 0);
}

} // namespace binary

/**
 * @namespace kaad::tensorfuncs::adjoint::unary
 */
namespace unary {

template <class Kernel>
using pointwise_fn = void (*)(const typename Kernel::value_type *A,
                              typename Kernel::value_type *dA,
                              const typename Kernel::value_type *C,
                              const typename Kernel::value_type *dC,
                              const typename Kernel::value_type *end);

template <typename T>
using sum_dim_fn = void (*)(T *dA, const T *dC, int *strideA, int *strideC,
                            size_t *a_dim_offsetset, int a_rank);

template <typename T>
using mean_fn = void (*)(T *dA, const T *dC, const T *dA_end, T divisor);

template <typename T>
using mean_dim_fn = void (*)(const T *A, T *dA, const T *C, const T *dC,
                             int *strideA, int *strideC,
                             size_t *a_dim_offsetset, int a_rank, T divisor,
                             const T *c_end);

template <typename T>
using slice_fn = void (*)(T *dA, const T *dC, int *strideA, int *strideC,
                          size_t *start_offset, size_t *c_dim_offset, int rank);

/**
 * @brief Accumulates the gradient of Op(A), A(tensor).
 * @pre @p C is a scalar.
 * @pre Every operand must have the same shape as their gradient.
 * @tparam Kernel A struct containing a static binary funcion ('Grad').
 * @param[in] A Pointer to the start of A(tensor).
 * @param[out] dA Pointer to the start of the gradient w.r.t. @p A.
 * @param[in] C Pointer to the start of C(scalar).
 * @param[in] dC Pointer to the start of the gradient w.r.t. @p C.
 * @param A_end Pointer to the end of @p A.
 */
template <class Kernel>
void scalarOut(const typename Kernel::value_type *A,
               typename Kernel::value_type *dA,
               const typename Kernel::value_type *C,
               const typename Kernel::value_type *dC,
               const typename Kernel::value_type *A_end) {
    for (; A != A_end; A++, dA++) {
        Kernel::Grad(*A, *dA, *C, *dC);
    }
}

/**
 * @brief Accumulates the gradient of Op(A), A(tensor).
 * @pre @p A and @p C have the same shape.
 * @pre Every operand must have the same shape as their gradient.
 * @tparam Kernel A struct containing a static binary funcion ('Grad').
 * @param[in] A Pointer to the start of A(tensor).
 * @param[out] dA Pointer to the start of the gradient w.r.t. @p A.
 * @param[in] C Pointer to the start of C(tensor).
 * @param[in] dC Pointer to the start of the gradient w.r.t. @p C.
 * @param C_end Pointer to the end of @p C.
 */
template <class Kernel>
void pointwise(const typename Kernel::value_type *A,
               typename Kernel::value_type *dA,
               const typename Kernel::value_type *C,
               const typename Kernel::value_type *dC,
               const typename Kernel::value_type *C_end) {
    for (; C != C_end; A++, dA++, C++, dC++) {
        Kernel::Grad(*A, *dA, *C, *dC);
    }
}

/**
 * @brief Accumulates the gradient of sum_dim(A), A(tensor).
 * @tparam T Element type
 * @param[out] dA Pointer to the start of the gradient w.r.t. @p A.
 * @param[in] dC Pointer to the start of the gradient w.r.t. @p C.
 * @param strideA Stride array for @p dA.
 * @param strideC Stride array for @p dC.
 * @param a_dim_offsetset Offset array per dimension
 * @param a_rank Number of dimensions
 */
template <typename T>
void sum_dim(T *dA, const T *dC, int *strideA, int *strideC,
             size_t *a_dim_offset, int a_rank) {
    const T *end = dA + *a_dim_offset;
    if (a_rank <= 1) {
        for (; dA != end; dA += *strideA, dC += *strideC) {
            *dA += *dC;
        }
    } else {
        for (; dA != end; dA += *strideA, dC += *strideC) {
            sum_dim(dA, dC, strideA + 1, strideC + 1, a_dim_offset + 1,
                    a_rank - 1);
        }
    }
}

/**
 * @brief Compile-time recursive version of runtime @ref sum_dim().
 */
template <typename T, int a_rank>
void sum_dim(T *dA, const T *dC, int *strideA, int *strideC,
             size_t *a_dim_offsetset, int _) {
    const T *end = dA + *a_dim_offsetset;
    if constexpr (a_rank <= 1) {
        for (; dA != end; dA += *strideA, dC += *strideC) {
            *dA += *dC;
        }
    } else {
        for (; dA != end; dA += *strideA, dC += *strideC) {
            sum_dim<T, a_rank - 1>(dA, dC, strideA + 1, strideC + 1,
                                   a_dim_offsetset + 1, 0);
        }
    }
}

/**
 * @brief Accumulates the gradient of mean_dim(A), A(tensor).
 * @tparam T Element type
 * @param[out] dA Pointer to the start of the gradient w.r.t. @p A.
 * @param[in] dC Pointer to the start of the gradient w.r.t. @p C.
 * @param dA_end Pointer to the end of @p dA.
 * @param divisor Length of dA.
 */
template <typename T>
void mean(T *dA, const T *dC, const T *dA_end, T divisor) {
    T mean = *dC / divisor;
    for (; dA != dA_end; dA++) {
        *dA += mean;
    }
}

/**
 * @brief Computes gradient of taking the mean of tensor A along a given
 * dimension.
 * @tparam T Element type
 * @param dA Pointer to the start of gradient tensor dA.
 * @param dC Pointer to the start of gradient tensor dC.
 * @param strideA Stride for dA
 * @param strideC Stride for dC
 * @param a_dim_offsetset Offset array per dimension
 * @param a_rank Number of dimensions
 * @param divisor divisor to compute mean of A (length of dimension summed over)
 * @param dA_end Pointer to the end of gradient tensor dA.
 */
/**
 * @brief Accumulates the gradient of mean_dim(A), A(tensor).
 * @tparam T Element type
 * @param[out] dA Pointer to the start of the gradient w.r.t. @p A.
 * @param[in] dC Pointer to the start of the gradient w.r.t. @p C.
 * @param strideA Stride array for @p dA.
 * @param strideC Stride array for @p dC.
 * @param a_dim_offsetset Offset array per dimension
 * @param a_rank Number of dimensions
 * @param divisor Length of relevant dimension.
 * @param dA_end Pointer to the end of @p dA.
 */
template <typename T>
void mean_dim(const T *A, T *dA, const T *C, const T *dC, int *strideA,
              int *strideC, size_t *a_dim_offset, int a_rank, T divisor,
              const T *dA_end) {
    sum_dim(dA, dC, strideA, strideC, a_dim_offset, a_rank);
    for (; dA != dA_end; dA++) {
        *dA /= divisor;
    }
}

/**
 * @brief Compile-time recursive version of runtime @ref mean_dim().
 */
template <typename T, int a_rank>
void mean_dim(const T *A, T *dA, const T *C, const T *dC, int *strideA,
              int *strideC, size_t *a_dim_offsetset, int _, T divisor,
              const T *dA_end) {
    sum_dim<T, a_rank>(dA, dC, strideA, strideC, a_dim_offsetset, 0);
    for (; dA != dA_end; dA++) {
        *dA /= divisor;
    }
}

/**
 * @brief Accumulates the gradient of slice(A), A(tensor).
 * @tparam T Element type
 * @param[out] dA Pointer to the start of the gradient w.r.t. @p A.
 * @param[in] dC Pointer to the start of the gradient w.r.t. @p C.
 * @param strideA Stride array for @p A.
 * @param strideC Stride array for @p C.
 * @param start_offset Offset for the start of C in A.
 * @param c_dim_offset Offset to the end of @p C per dimension.
 * @param rank Number of dimensions in A and C.
 */
template <typename T>
void slice(T *dA, const T *dC, int *strideA, int *strideC, size_t *start_offset,
           size_t *c_dim_offset, int rank) {
    dA += *start_offset;
    const T *end = dC + *c_dim_offset;
    if (rank <= 1) {
        for (; dC != end; dA += *strideA, dC += *strideC) {
            *dA = *dC;
        }
    } else {
        for (; dC < end; dA += *strideA, dC += *strideC) {
            slice(dA, dC, strideA + 1, strideC + 1, start_offset + 1,
                  c_dim_offset + 1, rank - 1);
        }
    }
}

/**
 * @brief Compile-time recursive version of runtime @ref slice().
 */
template <typename T, int rank>
void slice(T *dA, const T *dC, int *strideA, int *strideC, size_t *start_offset,
           size_t *c_dim_offset, int _) {
    dA += *start_offset;
    const T *end = dC + *c_dim_offset;
    if constexpr (rank <= 1) {
        for (; dC != end; dA += *strideA, dC += *strideC) {
            *dA = *dC;
        }
    } else {
        for (; dC < end; dA += *strideA, dC += *strideC) {
            slice<T, rank - 1>(dA, dC, strideA + 1, strideC + 1,
                               start_offset + 1, c_dim_offset + 1, 0);
        }
    }
}

} // namespace unary
} // namespace tensorfuncs::adjoint
} // namespace kaad
