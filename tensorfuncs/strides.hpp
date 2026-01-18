#pragma once

#include "../tensor/tensor.hpp" // for Tensor, Tensor_view
#include "../utils.hpp"         // for transp2D, combine_matrix
#include <cstddef>              // for size_t

namespace kaad {

template <typename T, class Kernel> struct Node_binary_flex;
template <typename T> struct Node_batch_matmul;
template <typename T> struct Node_matmul;
template <typename T> struct Node_mean_dim;
template <typename T> struct Node_sum_dim;
template <typename T> struct Node_slice;

/**
 * @namespace kaad::Strides
 * @brief Provides utilities for computing tensor traversal parameters.
 *
 * This namespace contains helper functions to set up stride and offset
 * arrays used in flexible tensor operations, particularly for broadcasting
 * and multi-dimensional iteration.
 */
namespace Strides {

/**
 * @brief Initializes stride and offset arrays for a binary_flex operation node.
 * @tparam T      The scalar type (e.g., float, double).
 * @tparam Kernel The kernel class providing Op and Grad.
 * @param node    The binary node where traversal metadata will be stored.
 */
template <typename T, class Kernel>
void flexible_binary(Node_binary_flex<T, Kernel> &node) {
    Tensor_view<T> A = node.A->value.view();
    Tensor_view<T> B = node.B->value.view();
    Tensor_view<T> C = node.value.view();

    node.C_nDims = C.nDims;
    node.strideA = new int[node.C_nDims];
    node.strideB = new int[node.C_nDims];
    node.strideC = new int[node.C_nDims];

    int idx, idxA, idxB, idxC;
    for (int i = 1; i <= node.C_nDims; i++) {
        idx = node.C_nDims - i;
        idxA = A.nDims - i;
        node.strideA[idx] = idxA >= 0 ? A.stride[idxA] : 0;
        idxB = B.nDims - i;
        node.strideB[idx] = idxB >= 0 ? B.stride[idxB] : 0;
        idxC = C.nDims - i;
        node.strideC[idx] = idxC >= 0 ? C.stride[idxC] : 0;
        // make sure strideC[idx] is 1 instead of 0 if C.shape[idx] is 1 for
        // traversing in flexible function
        if (node.strideC[idx] == 0 && C.shape[idxC] == 1) {
            node.strideC[idx] = 1;
        }
    }

    node.C_offset = new size_t[node.C_nDims];
    for (int i = 0; i < node.C_nDims; i++) {
        node.C_offset[i] = C.shape[i] * node.strideC[i];
    }
}

/**
 * @brief Prepares stride and shape parameters for a 2D matrix multiplication.
 * @tparam T        The scalar type (e.g., float, double).
 * @param A         View of the left-hand side matrix.
 * @param B         View of the right-hand side matrix.
 * @param C         View of the output matrix.
 * @param a_dim     Output variable for the number of rows in A.
 * @param b_dim     Output variable for the number of columns in B.
 * @param k         Output variable for the shared inner dimension.
 * @param strideA   Output stride array for A.
 * @param strideB   Output stride array for B.
 * @param strideC   Output stride array for C, modified for efficient traversal.
 */
template <typename T>
void matmul_impl(const Tensor_view<T> A, const Tensor_view<T> B,
                 const Tensor_view<T> C, int &a_dim, int &b_dim, int &k,
                 int *strideA, int *strideB, int *strideC) {
    a_dim = A.shape[0];
    b_dim = B.shape[1];
    k = A.shape[1];

    std::copy(A.stride, A.stride + 2, strideA);
    std::copy(B.stride, B.stride + 2, strideB);
    std::copy(C.stride, C.stride + 2, strideC);

    int idx, idxA, idxB, idxC;
    int offsetC = 0, prevC;
    for (int i = 1; i <= 2; i++) {
        idx = 2 - i;

        idxC = C.nDims - i;
        prevC = offsetC;
        offsetC += ((idxC >= 0 ? C.shape[idxC] : i) - 1) * strideC[idx];
        strideC[idx] -= prevC + strideC[idx + 1];
    }
}

/**
 * @brief Sets up metadata for computing a 2D matrix product and its gradients.
 * Prepares three `matmul_impl` passes: forward pass and two backward passes
 * (w.r.t. A and B).
 *
 * @tparam T     The scalar type.
 * @param A      Input tensor A.
 * @param B      Input tensor B.
 * @param node   Matrix multiplication node where traversal parameters are
 * stored.
 */
template <typename T> void matmul(Node_matmul<T> &node) {
    Tensor_view<T> A = node.A->value.view();
    Tensor_view<T> B = node.B->value.view();
    Tensor_view<T> C = node.value.view();

    int A_T_shape[2];
    int A_T_stride[2];
    transp2D(A.shape, A.stride, A.nDims, A_T_shape, A_T_stride);
    Tensor_view<T> A_T = A;
    A_T.shape = A_T_shape;
    A_T.stride = A_T_stride;

    int B_T_shape[2];
    int B_T_stride[2];
    transp2D(B.shape, B.stride, B.nDims, B_T_shape, B_T_stride);
    Tensor_view<T> B_T = B;
    B_T.shape = B_T_shape;
    B_T.stride = B_T_stride;

    matmul_impl(A, B, C, node.a_rows[0], node.b_cols[0], node.shared_dim[0],
                node.strideA, node.strideB, node.strideC);
    matmul_impl(C, B_T, A, node.a_rows[1], node.b_cols[1], node.shared_dim[1],
                node.strideC + 2, node.strideB + 2, node.strideA + 2);
    matmul_impl(A_T, C, B, node.a_rows[2], node.b_cols[2], node.shared_dim[2],
                node.strideA + 4, node.strideC + 4, node.strideB + 4);
}

/**
 * @brief Initializes stride and shape metadata for batched matrix
 * multiplication.
 * @tparam T       The scalar type.
 * @param A        View of input tensor A.
 * @param B        View of input tensor B.
 * @param C        View of output tensor C.
 * @param strideA  Output stride array for A.
 * @param strideB  Output stride array for B.
 * @param strideC  Output stride array for C.
 * @param c_shape  Output shape array for C.
 * @param a_off    Output offset for A in kernel loops.
 * @param b_off    Output offset for B in kernel loops.
 * @param k        Output inner dimension (A.cols == B.rows).
 * @param D        Output number of dimensions in broadcasted result.
 */
template <typename T>
void batch_matmul_impl(Tensor_view<T> &A, Tensor_view<T> &B, Tensor_view<T> &C,
                       int *&strideA, int *&strideB, int *&strideC,
                       int *&c_shape, int &a_off, int &b_off, int &k,
                       size_t &D) {
    a_off = A.stride[A.nDims - 1];
    b_off = B.stride[B.nDims - 2];
    k = A.shape[A.nDims - 1];

    D = std::max(A.nDims, B.nDims);
    c_shape = new int[D];

    combine_matrix(A.shape, A.nDims, B.shape, B.nDims, c_shape, D);

    strideA = new int[D];
    strideB = new int[D];
    strideC = new int[D];

    int idx, idxA, idxB, idxC;
    for (int i = 1; i <= D; i++) {
        idx = D - i;
        idxA = A.nDims - i;
        strideA[idx] = idxA >= 0 ? A.stride[idxA] : 0;
        idxB = B.nDims - i;
        strideB[idx] = idxB >= 0 ? B.stride[idxB] : 0;
        idxC = C.nDims - i;
        strideC[idx] = idxC >= 0 ? C.stride[idxC] : 0;
    }

    strideA[D - 1] = 0;
    strideB[D - 2] = 0;
}

/**
 * @brief Prepares metadata for batched matrix multiplication and gradients.
 * Prepares three `batch_matmul_impl` passes: forward pass and two backward
 * passes (w.r.t. A and B).
 *
 * @tparam T     The scalar type.
 * @param node   Batched matrix multiplication node with traversal metadata.
 */
template <typename T> void batch_matmul(Node_batch_matmul<T> &node) {
    Tensor_view<T> A = node.A->value.view();
    Tensor_view<T> B = node.B->value.view();
    Tensor_view<T> C = node.value.view();

    std::vector<int> a_T_shape(A.nDims);
    std::vector<int> a_T_stride(A.nDims);
    transp2D(A.shape, A.stride, A.nDims, a_T_shape.data(), a_T_stride.data());
    Tensor_view<T> a_T = A;
    a_T.shape = a_T_shape.data();
    a_T.stride = a_T_stride.data();

    std::vector<int> b_T_shape(B.nDims);
    std::vector<int> b_T_stride(B.nDims);
    transp2D(B.shape, B.stride, B.nDims, b_T_shape.data(), b_T_stride.data());
    Tensor_view<T> b_T = B;
    b_T.shape = b_T_shape.data();
    b_T.stride = b_T_stride.data();

    batch_matmul_impl(A, B, C, node.strideA[0], node.strideB[0],
                      node.strideC[0], node.c_shape[0], node.A_colStride[0],
                      node.B_rowStride[0], node.shared_dim[0], node.C_nDims);
    batch_matmul_impl(C, b_T, A, node.strideC[1], node.strideB[1],
                      node.strideA[1], node.c_shape[1], node.A_colStride[1],
                      node.B_rowStride[1], node.shared_dim[1], node.C_nDims);
    batch_matmul_impl(a_T, C, B, node.strideA[2], node.strideC[2],
                      node.strideB[2], node.c_shape[2], node.A_colStride[2],
                      node.B_rowStride[2], node.shared_dim[2], node.C_nDims);
}

/**
 * @brief Prepares stride and offset metadata for a generalized outer operation.
 * @tparam T       The scalar type (e.g., float, double).
 * @tparam Kernel  A functor or struct implementing the binary operation.
 * @param A        Input tensor A.
 * @param B        Input tensor B.
 * @param node     Binary flex node storing output tensor C and shape/stride
 * metadata.
 */
template <typename T, class Kernel>
void outer(Node_binary_flex<T, Kernel> &node) {
    Tensor_view<T> A = node.A->value.view();
    Tensor_view<T> B = node.B->value.view();
    Tensor_view<T> C = node.value.view();

    node.C_nDims = C.nDims;

    node.strideA = new int[node.C_nDims];
    node.strideB = new int[node.C_nDims];
    node.strideC = new int[node.C_nDims];

    std::copy(C.stride, C.stride + C.nDims, node.strideC);
    std::copy(A.stride, A.stride + A.nDims, node.strideA);
    std::copy(B.stride, B.stride + B.nDims, node.strideB + A.nDims);

    node.C_offset = new size_t[node.C_nDims];
    for (int i = 0; i < node.C_nDims; i++) {
        node.C_offset[i] = C.shape[i] * node.strideC[i];
    }
}

/**
 * @brief Computes stride and offset metadata for operations along a specific
 * tensor dimension.
 * @tparam T         The scalar type (e.g., float, double).
 * @param A          Input tensor.
 * @param C          Output tensor (e.g., reduced along `dim`).
 * @param dim        The dimension along which the operation is applied.
 * @param D          (out) Number of dimensions.
 * @param A_offset   (out) Array storing the maximum valid offset per dimension
 * for A.
 * @param strideA    (out) Stride array for A.
 * @param strideC    (out) Stride array for C, adjusted to zero along `dim`.
 */
template <typename T>
void along_dim_impl(Tensor_view<T> &A, Tensor_view<T> &C, int dim, size_t &D,
                    size_t *&A_offset, int *&strideA, int *&strideC) {
    D = A.nDims;
    strideA = new int[D];
    strideC = new int[D];

    std::copy(A.stride, A.stride + A.nDims, strideA);
    std::copy(A.stride, A.stride + A.nDims, strideC);
    // make sure stride[i] is 1 instead of 0 if shape[i] is 1 for
    // traversing in flexible function
    for (int i = 0; i < D; i++) {
        if (strideA[i] == 0 && A.shape[i] == 1) {
            strideA[i] = 1;
        }
    }

    strideC[dim] = 0;
    for (int i = 0; i < dim; i++) {
        strideC[i] /= A.shape[dim];
    }

    A_offset = new size_t[D];
    for (int i = 0; i < D; i++) {
        A_offset[i] = A.shape[i] * strideA[i];
    }
}

/**
 * @brief Initializes traversal metadata for a sum operation along a specified
 * dimension.
 * @tparam T     The scalar type.
 * @param A      Input tensor to be reduced.
 * @param node   Output node that stores the result and metadata.
 * @param dim    The dimension along which to compute the sum.
 */
template <typename T> void sum_dim(Node_sum_dim<T> &node, int dim) {
    Tensor_view<T> A = node.A->value.view();
    Tensor_view<T> C = node.value.view();

    along_dim_impl(A, C, dim, node.C_nDims, node.A_offset, node.strideA,
                   node.strideC);
}

/**
 * @brief Initializes traversal metadata for a mean operation along a specified
 * dimension.
 * @tparam T     The scalar type.
 * @param A      Input tensor.
 * @param node   Node to store the mean result and auxiliary metadata for
 * backward pass.
 * @param dim    The dimension along which to compute the mean.
 */
template <typename T> void mean_dim(Node_mean_dim<T> &node, int dim) {
    Tensor_view<T> A = node.A->value.view();
    Tensor_view<T> C = node.value.view();
    Tensor_view<T> dA = node.A->gradient.view();

    node.divisor = A.shape[dim];
    node.C_end = C.val + C.len;
    node.dA_end = dA.val + dA.len;

    along_dim_impl(A, C, dim, node.A_nDims, node.A_offset, node.strideA,
                   node.strideC);
}

/**
 * @brief Initializes traversal metadata for a tensor slicing operation.
 * @tparam T         The scalar type.
 * @param node       Node to store the sliced result and traversal metadata.
 * @param offset     Array of offsets specifying where slicing begins in each
 * dimension.
 */
template <typename T> void slice(Node_slice<T> &node, const int *offset) {
    Tensor_view<T> A = node.A->value.view();
    Tensor_view<T> C = node.value.view();

    node.C_nDims = C.nDims;
    node.strideA = new int[node.C_nDims];
    node.strideB = new int[node.C_nDims];
    node.strideC = new int[node.C_nDims];

    int idx, idxA, idxC;
    for (int i = 1; i <= node.C_nDims; i++) {
        idx = node.C_nDims - i;
        idxA = A.nDims - i;
        node.strideA[idx] = idxA >= 0 ? A.stride[idxA] : 0;
        idxC = C.nDims - i;
        node.strideC[idx] = idxC >= 0 ? C.stride[idxC] : 0;
        // make sure strideC[idx] is 1 instead of 0 if C.shape[idx] is 1 for
        // traversing in flexible function
        if (node.strideC[idx] == 0 && C.shape[idxC] == 1) {
            node.strideC[idx] = 1;
        }
    }

    node.C_offset = new size_t[node.C_nDims];
    for (int i = 0; i < node.C_nDims; i++) {
        node.C_offset[i] = C.shape[i] * node.strideC[i];
    }

    node.start_offset_a = new size_t[A.nDims];
    std::copy(offset, offset + A.nDims, node.start_offset_a);
    for (int i = 0; i < A.nDims; i++) {
        node.start_offset_a[i] *= node.strideA[i];
    }
}
}; // namespace Strides
} // namespace kaad
