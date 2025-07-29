#pragma once

#include "tensor.h" // for tView, transp2D, combine_matrix
#include <stddef.h> // for size_t

namespace kaad {

template <typename T, class Kernel> struct Node_binary_flex;
template <typename T> struct Node_batch_matmul;
template <typename T> struct Node_matmul;
template <typename T> struct Node_mean_dim;
template <typename T> struct Node_sum_dim;

namespace Strides {
template <typename T, class Kernel>
void flexible_binary(Tensor<T> &A, Tensor<T> &B,
                     Node_binary_flex<T, Kernel> &node) {
    Tensor<T> &C = node.value;

    node.D = C.nDims;
    node.strideA = new int[node.D];
    node.strideB = new int[node.D];
    node.strideC = new int[node.D];

    int idx, idxA, idxB, idxC;
    for (int i = 1; i <= node.D; i++) {
        idx = node.D - i;
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

    node.c_offset = new size_t[node.D];
    for (int i = 0; i < node.D; i++) {
        node.c_offset[i] = C.shape[i] * node.strideC[i];
    }
}

template <typename T>
void matmul_impl(tView<T> A, tView<T> B, tView<T> C, int &a_dim, int &b_dim,
                 int &k, int *strideA, int *strideB, int *strideC) {
    a_dim = A.shape[0];
    b_dim = B.shape[1];
    k = A.shape[1];

    std::copy(A.stride, A.stride + 2, strideA);
    std::copy(B.stride, B.stride + 2, strideB);
    std::copy(C.stride, C.stride + 2, strideC);

    int idx, idxA, idxB, idxC;
    int offsetA = 0, _offsetA, offsetB = 0, _offsetB, offsetC = 0, _offsetC;
    for (int i = 1; i <= 2; i++) {
        idx = 2 - i;

        idxC = C.nDims - i;
        _offsetC = offsetC;
        offsetC += ((idxC >= 0 ? C.shape[idxC] : i) - 1) * strideC[idx];
        strideC[idx] -= _offsetC + strideC[idx + 1];
    }
}

template <typename T>
void matmul(Tensor<T> &A, Tensor<T> &B, Node_matmul<T> &node) {
    tView<T> a = A.view();
    tView<T> b = B.view();
    tView<T> c = node.value.view();

    int shapeBlock[8];
    tView<T> A_T = A.view();
    A_T.shape = shapeBlock;
    A_T.stride = shapeBlock + 2;
    transp2D(a.shape, a.stride, a.nDims, A_T.shape, A_T.stride);
    tView<T> B_T = B.view();
    B_T.shape = shapeBlock + 4;
    B_T.stride = shapeBlock + 6;
    transp2D(b.shape, b.stride, b.nDims, B_T.shape, B_T.stride);

    matmul_impl(a, b, c, node.a_dim[0], node.b_dim[0], node.k[0], node.strideA,
                node.strideB, node.strideC);
    matmul_impl(c, B_T, a, node.a_dim[1], node.b_dim[1], node.k[1],
                node.strideC + 2, node.strideB + 2, node.strideA + 2);
    matmul_impl(A_T, c, b, node.a_dim[2], node.b_dim[2], node.k[2],
                node.strideA + 4, node.strideC + 4, node.strideB + 4);
}

template <typename T>
void batch_matmul_impl(tView<T> &A, tView<T> &B, tView<T> &C, int *&strideA,
                       int *&strideB, int *&strideC, int *&c_shape, int &a_off,
                       int &b_off, int &k, size_t &D) {
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

template <typename T>
void batch_matmul(Tensor<T> &A, Tensor<T> &B, Node_batch_matmul<T> &node) {
    tView<T> a = A.view();
    tView<T> b = B.view();
    tView<T> c = node.value.view();
    int *shapeBlock = new int[B.nDims * 2 + A.nDims * 2];
    tView<T> a_T = A.view();
    a_T.shape = shapeBlock;
    a_T.stride = a_T.shape + A.nDims;
    transp2D(A.shape, A.stride, A.nDims, a_T.shape, a_T.stride);
    tView<T> b_T = B.view();
    b_T.shape = a_T.stride + A.nDims;
    b_T.stride = b_T.shape + B.nDims;
    transp2D(B.shape, B.stride, B.nDims, b_T.shape, b_T.stride);

    batch_matmul_impl(a, b, c, node.strideA[0], node.strideB[0],
                      node.strideC[0], node.c_shape[0], node.a_offset[0],
                      node.b_offset[0], node.k[0], node.D);
    batch_matmul_impl(c, b_T, a, node.strideC[1], node.strideB[1],
                      node.strideA[1], node.c_shape[1], node.a_offset[1],
                      node.b_offset[1], node.k[1], node.D);
    batch_matmul_impl(a_T, c, b, node.strideA[2], node.strideC[2],
                      node.strideB[2], node.c_shape[2], node.a_offset[2],
                      node.b_offset[2], node.k[2], node.D);

    delete[] shapeBlock;
}

template <typename T, class Kernel>
void outer(Tensor<T> &A, Tensor<T> &B, Node_binary_flex<T, Kernel> &node) {
    Tensor<T> &C = node.value;

    node.D = C.nDims;
    node.reps = new int[node.D];
    std::copy(C.shape, C.shape + C.nDims, node.reps);
    for (int i = 0; i < node.D; i++) {
        node.reps[i]--;
    }

    node.count = new int[node.D];
    std::copy(node.reps, node.reps + node.D, node.count);

    node.strideA = new int[node.D];
    node.strideB = new int[node.D];
    node.strideC = new int[node.D];

    std::copy(C.stride, C.stride + C.nDims, node.strideC);
    std::copy(A.stride, A.stride + A.nDims, node.strideA);
    std::copy(B.stride, B.stride + B.nDims, node.strideB + A.nDims);

    // pad A.shape with 1s on the right
    int *a_shape_big = new int[node.D];
    std::copy(A.shape, A.shape + A.nDims, a_shape_big);
    a_shape_big[A.nDims] = 1;

    int idx, idxA, idxB, idxC;
    int offsetA = 0, _offsetA, offsetB = 0, _offsetB, offsetC = 0, _offsetC;
    for (int i = 1; i <= node.D; i++) {
        idx = node.D - i;

        _offsetA = offsetA;
        offsetA += (a_shape_big[idx] - 1) * node.strideA[idx];
        node.strideA[idx] -= _offsetA;

        idxB = B.nDims - i;
        _offsetB = offsetB;
        offsetB += ((idxB >= 0 ? B.shape[idxB] : 1) - 1) * node.strideB[idx];
        node.strideB[idx] -= _offsetB;

        idxC = C.nDims - i;
        _offsetC = offsetC;
        offsetC += ((idxC >= 0 ? C.shape[idxC] : 1) - 1) * node.strideC[idx];
        node.strideC[idx] -= _offsetC;
    }
}

template <typename T>
void along_dim_impl(Tensor<T> &A, Tensor<T> &C, int dim, size_t &D,
                    size_t *&a_offset, int *&strideA, int *&strideC) {
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

    a_offset = new size_t[D];
    for (int i = 0; i < D; i++) {
        a_offset[i] = A.shape[i] * strideA[i];
    }
}

template <typename T>
void sum_dim(Tensor<T> &A, Node_sum_dim<T> &node, int dim) {
    Tensor<T> &C = node.value;

    along_dim_impl(A, C, dim, node.D, node.a_offset, node.strideA,
                   node.strideC);
}

template <typename T>
void mean_dim(Tensor<T> &A, Node_mean_dim<T> &node, int dim) {
    Tensor<T> &C = node.value;
    Tensor<T> &dA = node.in1->gradient;
    node.divisor = A.shape[dim];
    node.c_end = C.val + C.len;
    node.dA_end = dA.val + dA.len;

    along_dim_impl(A, C, dim, node.D, node.a_offset, node.strideA,
                   node.strideC);
}
}; // namespace Strides
} // namespace kaad
