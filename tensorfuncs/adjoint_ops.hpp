#pragma once

#include "primal_ops.hpp" // for batch_matmul, matmul
#include <cstddef>        // for size_t

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

/**
 * @brief Computes gradients for an elementwise operation with a scalar
 * right-hand side.
 *
 * Applies the binary gradient function `grad` to compute the derivatives of a
 * scalar-tensor operation where the right-hand side is a scalar (B).
 *
 * Computation performed: grad(A[i], dA[i], B[0], dB[0], C[i], dC[i]);
 *
 * @tparam T Element type
 * @tparam Grad A callable object that computes the gradient.
 *
 * @param A Pointer to the start of input tensor A.
 * @param dA Pointer to the start of the gradient tensor dA.
 * @param B Pointer to the scalar input B (shape = 1).
 * @param dB Pointer to the gradient of the scalar B (shape = 1).
 * @param C Pointer to the start of the output tensor C.
 * @param dC Pointer to the start of the gradient tensor dC.
 * @param C_end Pointer to the end of the output tensor C.
 * @param grad A callable that computes the gradients.
 */
template <typename T, class Grad>
void scalarRhs(const T *A, T *dA, const T *B, T *dB, const T *C, const T *dC,
               const T *C_end, Grad grad) {
    for (; C != C_end; A++, dA++, C++, dC++) {
        grad(*A, *dA, *B, *dB, *C, *dC);
    }
}

/**
 * @brief Computes gradients for an elementwise operation with a scalar
 * left-hand side.
 *
 * Applies the binary gradient function `grad` to compute the derivatives of a
 * scalar-tensor operation where the left-hand side is a scalar (A).
 *
 * Computation performed: grad(A[0], dA[0], B[i], dB[i], C[i], dC[i]);
 *
 * @tparam T    Element type
 * @tparam Grad A callable object that computes the gradient.
 *
 * @param A Pointer to the scalar input A (shape = 1).
 * @param dA Pointer to the gradient of the scalar A (shape = 1).
 * @param B Pointer to the start of input tensor B.
 * @param dB Pointer to the start of the gradient tensor dB.
 * @param C Pointer to the start of the output tensor C.
 * @param dC Pointer to the start of the gradient tensor dC.
 * @param C_end Pointer to the end of the output tensor C.
 * @param grad A callable that computes the gradients.
 */
template <typename T, class Grad>
void scalarLhs(const T *A, T *dA, const T *B, T *dB, const T *C, const T *dC,
               const T *C_end, Grad grad) {
    for (; C != C_end; B++, dB++, C++, dC++) {
        grad(*A, *dA, *B, *dB, *C, *dC);
    }
}

/**
 * @brief Computes gradients for an elementwise operation with two tensors
 *
 * Applies the binary gradient function `grad` to compute the derivatives of a
 * tensor operation where both tensors have the same shape and strides
 *
 * Computation performed: grad(A[i], dA[i], B[i], dB[i], C[i], dC[i]);
 *
 * @tparam T The Element type
 * @tparam Grad A callable object that computes the gradient.
 *
 * @param A Pointer to the start of input tensor A.
 * @param dA Pointer to the start of the gradient tensor dA.
 * @param B Pointer to the start of input tensor B.
 * @param dB Pointer to the start of the gradient tensor dB.
 * @param C Pointer to the start of the output tensor C.
 * @param dC Pointer to the start of the gradient tensor dC.
 * @param C_end Pointer to the end of the output tensor C.
 * @param grad A callable that computes the gradients.
 */
template <typename T, class Grad>
void pointwise(const T *A, T *dA, const T *B, T *dB, const T *C, const T *dC,
               const T *C_end, Grad grad) {
    for (; C != C_end; A++, dA++, B++, dB++, C++, dC++) {
        grad(*A, *dA, *B, *dB, *C, *dC);
    }
}

/**
 * @brief Computes gradients for a broadcastable elementwise operation with
 * arbitrary strides.
 *
 * Recursively applies the binary gradient function `grad` to compute the
 * derivatives of a tensor operation where A, B, and C may have different shapes
 * and strides.
 *
 * For N = 1 (innermost dimension), computes:
 *   grad(A[i], dA[i], B[i], dB[i], C[i], dC[i]);
 * using the given strides.
 *
 * For N > 1, the function recursively traverses the outer dimensions.
 *
 * @tparam T Element type
 * @tparam Grad A callable object that computes the gradient.
 *
 * @param A Pointer to the start of input tensor A.
 * @param dA Pointer to the start of the gradient tensor dA.
 * @param B Pointer to the start of input tensor B.
 * @param dB Pointer to the start of the gradient tensor dB.
 * @param C Pointer to the start of output tensor C.
 * @param dC Pointer to the start of the gradient tensor dC.
 * @param strideA Stride array for A.
 * @param strideB Stride array for B.
 * @param strideC Stride array for C.
 * @param c_offset Total number of elements and per-dim offsets of C.
 * @param N Number of dimensions.
 * @param grad A callable that computes the gradients.
 */
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

/**
 * @brief Compile-time recursive version of flexible().
 * @see flexible(const T *A, T *dA, const T *B, T *dB, const T *C, const T *dC,
 * int *strideA, int *strideB, int *strideC, size_t *c_offset, int N, Grad grad)
 */
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

/**
 * @brief Computes gradients for the dot product of a vector and a scalar.
 *
 * Computation performed: dA[i] += dC[0] * B[0]
 *                        dB[0] += dC[0] * A[i]
 *
 * @tparam T    Element type
 * @tparam Grad Dummy type for consistency.
 *
 * @param A Pointer to the beginning of vector A.
 * @param dA Pointer to the beginning of gradient vector A.
 * @param B Pointer to input scalar B.
 * @param dB Pointer to gradient scalar B.
 * @param C Pointer to scalar result C.
 * @param dC Pointer to scalar gradient result C.
 * @param A_end Pointer to the end of the input vector A.
 * @param _ Ignored gradient type.
 */
template <typename T, class Grad>
void scalarDot(const T *A, T *dA, const T *B, T *dB, const T *C, const T *dC,
               const T *A_end, Grad _) {
    for (; A != A_end; A++, dA++) {
        *dA += *dC * (*B);
        *dB += *dC * (*A);
    }
}

/**
 * @brief Computes gradients for the dot product of two vectors.
 *
 * Computation performed: dA[i] += dC[0] * B[i]
 *                        dB[i] += dC[0] * A[i]
 *
 * @tparam T    Element type
 * @tparam Grad Dummy type for consistency.
 *
 * @param A Pointer to the beginning of vector A.
 * @param dA Pointer to the beginning of gradient vector A.
 * @param B Pointer to the beginning of vector B.
 * @param dB Pointer to the beginning of gradient vector B.
 * @param C Pointer to scalar result C.
 * @param dC Pointer to scalar gradient result C.
 * @param A_end Pointer to the end of the input vector A.
 * @param _ Ignored gradient type.
 */
template <typename T, class Grad>
void dot(const T *A, T *dA, const T *B, T *dB, const T *C, const T *dC,
         const T *A_end, Grad _) {
    for (; A != A_end; A++, dA++, B++, dB++) {
        *dA += *dC * (*B);
        *dB += *dC * (*A);
    }
}

/**
 * @brief Computest the Gradient of a matrix multiplication.
 *
 * Computation performed: dA += dC × Bᵀ
 *                        dB += dC × Aᵀ
 *
 * @tparam T Value type.
 *
 * @param A Pointer to input matrix A.
 * @param dA Pointer to gradient matrix A.
 * @param B Pointer to input matrix B.
 * @param dB Pointer to gradient matrix B.
 * @param C Pointer to result matrix C.
 * @param dC Pointer to gradient matrix C.
 * @param a_dim Rows of A.
 * @param b_dim Columns of B.
 * @param k Shared inner dimension (A.cols == B.rows).
 * @param strideA Strides for matrix A.
 * @param strideB Strides for matrix B.
 * @param strideC Strides for matrix C.
 */
template <typename T>
void matmul(const T *A, T *dA, const T *B, T *dB, const T *C, const T *dC,
            int *a_dim, int *b_dim, int *k, int *strideA, int *strideB,
            int *strideC) {
    // dA = dC * B^T
    tensorfuncs::primal::binary::matmul(dC, B, dA, a_dim[0], b_dim[0], k[0],
                                        strideC, strideB, strideA);
    // dB = A^T * dC
    tensorfuncs::primal::binary::matmul(A, dC, dB, a_dim[1], b_dim[1], k[1],
                                        strideA + 2, strideC + 2, strideB + 2);
}

/**
 * @brief Computest the Gradient of a batched matrix multiplication.
 *
 * Computation performed: dA += dC × Bᵀ
 *                        dB += dC × Aᵀ
 *
 * @tparam T Value type.
 *
 * @param A Pointer to the start of input tensor A.
 * @param dA Pointer to the start of gradient tensor dA.
 * @param B Pointer to the start of input tensor B.
 * @param dB Pointer to the start of gradient tensor dB.
 * @param C Pointer to the start of output tensor C.
 * @param dC Pointer to the start of gradient tensor dC.
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
void batch_matmul(const T *A, T *dA, const T *B, T *dB, const T *C, const T *dC,
                  int **strideA, int **strideB, int **strideC, int **c_shape,
                  int *a_off, int *b_off, int *k, int N) {
    // dA = dC * B^T
    tensorfuncs::primal::binary::batch_matmul<T>(
        dC, B, dA, strideC[0], strideB[0], strideA[0], c_shape[0], a_off[0],
        b_off[0], k[0], N);
    // dB = A^T * dC
    tensorfuncs::primal::binary::batch_matmul<T>(
        A, dC, dB, strideA[1], strideC[1], strideB[1], c_shape[1], a_off[1],
        b_off[1], k[1], N);
}

/**
 * @brief Compile-time recursive version of batched matrix multiplication
 * gradient.
 * @see batch_matmul(const T *A, T *dA, const T *B, T *dB, const T *C, const T
 * *dC, int **strideA, int **strideB, int **strideC, int **c_shape, int *a_off,
 * int *b_off, int *k, int N)
 */
template <typename T, int N>
void batch_matmul(const T *A, T *dA, const T *B, T *dB, const T *C, const T *dC,
                  int **strideA, int **strideB, int **strideC, int **c_shape,
                  int *a_off, int *b_off, int *k, int _) {
    // dA = dC * B^T
    tensorfuncs::primal::binary::batch_matmul<T, N>(
        dC, B, dA, strideC[0], strideB[0], strideA[0], c_shape[0], a_off[0],
        b_off[0], k[0], 0);
    // dB = A^T * dC
    tensorfuncs::primal::binary::batch_matmul<T, N>(
        A, dC, dB, strideA[1], strideC[1], strideB[1], c_shape[1], a_off[1],
        b_off[1], k[1], 0);
}

} // namespace binary

/**
 * @namespace kaad::tensorfuncs::adjoint::unary
 */
namespace unary {

/**
 * @brief Computes gradients for an elementwise operation with a scalar
 * output.
 *
 * Applies the unary gradient function `grad` to compute the derivatives of a
 * scalar-tensor operation where the output is a scalar (C).
 *
 * Computation performed: grad(A[i], dA[i], C[0], dC[0]);
 *
 * @tparam T    Element type
 * @tparam Grad A callable object that computes the gradient.
 *
 * @param A     Pointer to the start of input tensor A.
 * @param dA    Pointer to the start of the gradient tensor dA.
 * @param C     Pointer to the start of the output tensor C.
 * @param dC    Pointer to the start of the gradient tensor dC.
 * @param A_end Pointer to the end of input tensor A.
 * @param grad  A callable that computes the gradients.
 */
template <typename T, class Grad>
void scalarOut(const T *A, T *dA, const T *C, const T *dC, const T *A_end,
               Grad grad) {
    for (; A != A_end; A++, dA++) {
        grad(*A, *dA, *C, *dC);
    }
}

/**
 * @brief Computes gradients for an elementwise operation with a tensor.
 *
 * Applies the binary gradient function `grad` to compute the derivatives of a
 * tensor operation.
 *
 * Computation performed: grad(A[i], dA[i], C[i], dC[i]);
 *
 * @tparam T    Element type
 * @tparam Grad A callable object that computes the gradient.
 *
 * @param A     Pointer to the start of input tensor A.
 * @param dA    Pointer to the start of the gradient tensor dA.
 * @param C     Pointer to the start of the output tensor C.
 * @param dC    Pointer to the start of the gradient tensor dC.
 * @param C_end Pointer to the end of the output tensor C.
 * @param grad  A callable that computes the gradients.
 */
template <typename T, class Grad>
void pointwise(const T *A, T *dA, const T *C, const T *dC, const T *C_end,
               Grad grad) {
    for (; C != C_end; A++, dA++, C++, dC++) {
        grad(*A, *dA, *C, *dC);
    }
}

/**
 * @brief Computes gradient of summing a tensor A along a given dimension.
 * @tparam T Element type
 * @param dA Pointer to the start of gradient tensor dA.
 * @param dC Pointer to the start of gradient tensor dC.
 * @param strideA Stride for dA
 * @param strideC Stride for dC
 * @param a_offset Offset array per dimension
 * @param N Number of dimensions
 */
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

/**
 * @brief Compile-time recursive version of gradient of sum of a tensor along a
 * given dimension gradient.
 * @see sum_dim(T *dA, const T *dC, int *strideA, int *strideC, size_t
 * *a_offset, int N)
 */
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

/**
 * @brief Computes gradient of taking the mean of all Elements of A into C
 * @tparam T Element type
 * @param dA Pointer to the start of gradient tensor dA.
 * @param dC Pointer to the gradient scalar dC.
 * @param dA_end Pointer to the end of gradient tensor dA.
 * @param divisor Divisor for distributing dC over dA.
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
 * @param a_offset Offset array per dimension
 * @param N Number of dimensions
 * @param divisor divisor to compute mean of A (length of dimension summed over)
 * @param dA_end Pointer to the end of gradient tensor dA.
 */
template <typename T>
void mean_dim(const T *A, T *dA, const T *C, T *dC, int *strideA, int *strideC,
              size_t *a_offset, int N, T divisor, T *dA_end) {
    sum_dim(dA, dC, strideA, strideC, a_offset, N);
    for (; dA != dA_end; dA++) {
        *dA /= divisor;
    }
}

/**
 * @brief Compile-time recursive version of gradient of mean of a tensor along a
 * given dimension
 * @see mean_dim(const T *A, T *dA, const T *C, T *dC, int *strideA, int
 * *strideC, size_t *a_offset, int N, T divisor, T *dA_end)
 */
template <typename T, int N>
void mean_dim(const T *A, T *dA, const T *C, T *dC, int *strideA, int *strideC,
              size_t *a_offset, int _, T divisor, T *dA_end) {
    sum_dim<T, N>(dA, dC, strideA, strideC, a_offset, 0);
    for (; dA != dA_end; dA++) {
        *dA /= divisor;
    }
}

/**
 * @brief Computes the gradient of copyine a sliced view of A into C.
 * @tparam T Element type
 * @param dA Pointer to the start of gradient tensor dA.
 * @param dC Pointer to the start of gradient tensor dC.
 * @param strideA Stride for A
 * @param strideC Stride for C
 * @param start_offset Offset to apply to A
 * @param c_offset Size of output slice
 * @param N Number of dimensions
 */
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

/**
 * @brief Compile-time recursive version of slice.
 * @see slice(const T *A, T *C, int *strideA, int *strideC, size_t
 * *start_offset, size_t *c_offset, int N)
 */
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
} // namespace tensorfuncs::adjoint
} // namespace kaad
