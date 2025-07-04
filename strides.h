#include <stddef.h>   // for size_t
#include <algorithm>  // for copy
#include "tensor.h"   // for Tensor (ptr only), tView, transp2D, combine_matrix
template <typename T, class Kernel> struct Node_binary_flex;
template <typename T, class Kernel> struct Node_unary_flex;
template <typename T> struct Node_batch_matmul;
template <typename T> struct Node_matmul;
template <typename T> struct Node_mean_dim;

#pragma once

template<typename T>
struct Strides {
    template <class Kernel>
    static void flexible_binary(Tensor<T>& A, Tensor<T>& B, Node_binary_flex<T,Kernel>& node) {
        Tensor<T>& C = node.value;

        node.D = C.nDims;
        node.reps = new int[node.D];
        copy(C.shape, C.shape + C.nDims, node.reps);
        for (int i = 0; i < node.D; i++) {
            node.reps[i]--;
        }

        node.count = new int[node.D];
        copy(node.reps, node.reps + node.D, node.count);

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
        }
    
        int offsetA = 0, _offsetA, offsetB = 0, _offsetB, offsetC = 0, _offsetC;
        for (int i = 1; i <= node.D; i++) {
            idx = node.D - i;

            idxA = A.nDims - i;
            _offsetA = offsetA;
            offsetA += ((idxA >= 0 ? A.shape[idxA] : i) - 1) * node.strideA[idx];
            node.strideA[idx] -= _offsetA;

            idxB = B.nDims - i;
            _offsetB = offsetB;
            offsetB += ((idxB >= 0 ? B.shape[idxB] : i) - 1) * node.strideB[idx];
            node.strideB[idx] -= _offsetB;

            idxC = C.nDims - i;
            _offsetC = offsetC;
            offsetC += ((idxC >= 0 ? C.shape[idxC] : i) - 1) * node.strideC[idx];
            node.strideC[idx] -= _offsetC;
        }
    }

    static void matmul(Tensor<T>& A, Tensor<T>& B, Node_matmul<T>& node) {
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

        _matmul(a, b, c, node.a_dim[0], node.b_dim[0], node.k[0], node.strideA, node.strideB, node.strideC);
        _matmul(c, B_T, a, node.a_dim[1], node.b_dim[1], node.k[1], node.strideC+2, node.strideB+2, node.strideA+2);
        _matmul(A_T, c, b, node.a_dim[2], node.b_dim[2], node.k[2], node.strideA+4, node.strideC+4, node.strideB+4);
    }

    static void batch_matmul(Tensor<T>& A, Tensor<T>& B, Node_batch_matmul<T>& node) {
        tView<T> a = A.view();
        tView<T> b = B.view();
        tView<T> c = node.value.view();
        int* shapeBlock = new int[B.nDims * 2 + A.nDims * 2];
        tView<T> a_T = A.view();
        a_T.shape = shapeBlock;
        a_T.stride = a_T.shape + A.nDims;
        transp2D(A.shape, A.stride, A.nDims, a_T.shape, a_T.stride);
        tView<T> b_T = B.view();
        b_T.shape = a_T.stride + A.nDims;
        b_T.stride = b_T.shape + B.nDims;
        transp2D(B.shape, B.stride, B.nDims, b_T.shape, b_T.stride);

        _batch_matmul(a, b, c, node.a_offset[0], node.b_offset[0], node.k[0], node.D[0], node.reps[0], node.count[0], node.strideA[0], node.strideB[0], node.strideC[0]); 
        _batch_matmul(c, b_T, a, node.a_offset[1], node.b_offset[1], node.k[1], node.D[1], node.reps[1], node.count[1], node.strideC[1], node.strideB[1], node.strideA[1]); 
        _batch_matmul(a_T, c, b, node.a_offset[2], node.b_offset[2], node.k[2], node.D[2], node.reps[2], node.count[2], node.strideA[2], node.strideC[2], node.strideB[2]); 

        delete[] shapeBlock;
    }

    template <class Kernel>
    static void outer(Tensor<T>& A, Tensor<T>& B, Node_binary_flex<T,Kernel>& node) {
        Tensor<T>& C = node.value;

        node.D = C.nDims;
        node.reps = new int[node.D];
        copy(C.shape, C.shape + C.nDims, node.reps);
        for (int i = 0; i < node.D; i++) {
            node.reps[i]--;
        }

        node.count = new int[node.D];
        copy(node.reps, node.reps + node.D, node.count);

        node.strideA = new int[node.D];
        node.strideB = new int[node.D];
        node.strideC = new int[node.D];

        copy(C.stride, C.stride + C.nDims, node.strideC);
        copy(A.stride, A.stride + A.nDims, node.strideA);
        copy(B.stride, B.stride + B.nDims, node.strideB + A.nDims);

        // pad A.shape with 1s on the right
        int* a_shape_big = new int[node.D];
        copy(A.shape, A.shape + A.nDims, a_shape_big);
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

    template <class Kernel>
    static void along_dim(Tensor<T>& A, Node_unary_flex<T,Kernel>& node, int dim) {
        Tensor<T>& C = node.value;

        _along_dim(A, C, dim, node.D, node.reps, node.count, node.strideA, node.strideC);
    }

    static void mean_along_dim(Tensor<T>& A, Node_mean_dim<T>& node, int dim) {
        Tensor<T>& C = node.value;

        node.divisor = (T)A.shape[dim];
        node.c_len[0] = C.len;
        node.c_len[1] = A.len;

        _along_dim(A, C, dim, node.D, node.reps, node.count, node.strideA, node.strideC);
    }

    private:

    static void _matmul(tView<T> A, tView<T> B, tView<T> C, int& a_dim, int& b_dim, int& k, int* strideA, int* strideB, int* strideC) {
        a_dim = A.shape[0];
        b_dim = B.shape[1];
        k = A.shape[1];

        copy(A.stride, A.stride + 2, strideA);
        copy(B.stride, B.stride + 2, strideB);
        copy(C.stride, C.stride + 2, strideC);

        int idx, idxA, idxB, idxC;
        int offsetA = 0, _offsetA, offsetB = 0, _offsetB, offsetC = 0, _offsetC;
        for (int i = 1; i <= 2; i++) {
            idx = 2 - i;

            //idxA = A.nDims - i;
            //_offsetA = offsetA;
            //offsetA += ((idxA >= 0 ? A.shape[idxA] : i) - 1) * strideA[idx];
            //strideA[idx] -= _offsetA;

            //idxB = B.nDims - i;
            //_offsetB = offsetB;
            //offsetB += ((idxB >= 0 ? B.shape[idxB] : i) - 1) * strideB[idx];
            //strideB[idx] -= _offsetB;

            idxC = C.nDims - i;
            _offsetC = offsetC;
            offsetC += ((idxC >= 0 ? C.shape[idxC] : i) - 1) * strideC[idx];
            strideC[idx] -= _offsetC + strideC[idx + 1];
        }
    }
    
    static void _batch_matmul(tView<T>& A, tView<T>& B, tView<T>& C, int& a_off, int& b_off, int& k, size_t& D, int*& reps, int*& count, int*& strideA, int*& strideB, int*& strideC) {
        a_off = A.stride[A.nDims - 1];
        b_off = B.stride[B.nDims - 2];
        k = A.shape[A.nDims - 1];

        D = max(A.nDims, B.nDims);
        reps = new int[D];

        combine_matrix(A.shape, A.nDims, B.shape, B.nDims, reps, D);
        for (int i = 0; i < D; i++) {
            reps[i]--;
        }

        count = new int[D];
        copy(reps, reps + D, count);

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
        int offsetA = 0, _offsetA, offsetB = 0, _offsetB, offsetC = 0, _offsetC;
        for (int i = 1; i <= D; i++) {
            idx = D - i;

            idxA = A.nDims - i;
            _offsetA = offsetA;
            offsetA += ((idxA >= 0 ? A.shape[idxA] : i) - 1) * strideA[idx];
            strideA[idx] -= _offsetA;

            idxB = B.nDims - i;
            _offsetB = offsetB;
            offsetB += ((idxB >= 0 ? B.shape[idxB] : i) - 1) * strideB[idx];
            strideB[idx] -= _offsetB;

            idxC = C.nDims - i;
            _offsetC = offsetC;
            offsetC += ((idxC >= 0 ? C.shape[idxC] : i) - 1) * strideC[idx];
            strideC[idx] -= _offsetC;
        }
    }

    static void _along_dim(Tensor<T>& A, Tensor<T>& C, int dim, size_t& D, int*& reps, int*& count, int*& strideA, int*& strideC) {
        D = A.nDims;
        reps = new int[D];
        copy(A.shape, A.shape + A.nDims, reps);
        for (int i = 0; i < D; i++) {
            reps[i]--;
        }

        count = new int[D];
        copy(reps, reps + D, count);

        strideA = new int[D];
        strideC = new int[D];

        copy(A.stride, A.stride + A.nDims, strideA);
        copy(A.stride, A.stride + A.nDims, strideC);
        strideC[dim] = 0;
        for (int i = 0; i < dim; i++) {
            strideC[i] /= A.shape[dim];
        }

        // insert 1 at C.shape[dim];
        int* c_shape_big = new int[A.nDims];
        copy(C.shape, C.shape + dim, c_shape_big);
        c_shape_big[dim] = 1;
        copy(C.shape + dim, C.shape + C.nDims, c_shape_big + dim + 1);

        int idx, idxC;
        int offsetA = 0, _offsetA, offsetC = 0, _offsetC;
        for (int i = 1; i <= D; i++) {
            idx = D - i;

            _offsetA = offsetA;
            offsetA += ((idx >= 0 ? A.shape[idx] : i) - 1) * strideA[idx];
            strideA[idx] -= _offsetA;

            _offsetC = offsetC;
            offsetC += (c_shape_big[idx] - 1) * strideC[idx];
            strideC[idx] -= _offsetC;
        }
    }
};
