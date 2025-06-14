#pragma once

#include "tensor.h"
#include "recorder.h"
#include "operations.h"

template<typename T>
struct Strides {
    static void pointwise(Node<T>& node) {
        node.nEntries = 2;
        node.strideLen = new size_t[2] { node.value.len, node.value.len };
        node.reps = new int*[2];
        node.count = new int*[2];
        node.strideA = new int*[2];
        node.strideB = new int*[2];
        node.strideC = new int*[2];
    }

    static void iterOverInp(Tensor<T>& A, Tensor<T>& B, Node<T>& node) {
        node.nEntries = 2;
        size_t len = max(A.len, B.len);
        node.strideLen = new size_t[2] { len, len };
        node.reps = new int*[2];
        node.count = new int*[2];
        node.strideA = new int*[2];
        node.strideB = new int*[2];
        node.strideC = new int*[2];
    }

    static void flexible(Tensor<T>& A, Tensor<T>& B, Node<T>& node) {
        Tensor<T>& C = node.value;

        node.nEntries = 2;
        node.strideLen = new size_t[node.nEntries];
        node.reps = new int*[node.nEntries];
        node.count = new int*[node.nEntries];
        node.strideA = new int*[node.nEntries];
        node.strideB = new int*[node.nEntries];
        node.strideC = new int*[node.nEntries];

        _flexible(A, B, C, node.strideLen[0], node.reps[0], node.count[0], node.strideA[0], node.strideB[0], node.strideC[0]);

        node.strideLen[1] = node.strideLen[0];

        node.reps[1] = new int[node.strideLen[0]];
        copy(node.reps[0], node.reps[0] + node.strideLen[0], node.reps[1]);

        node.count[1] = new int[node.strideLen[0]];
        copy(node.count[0], node.count[0] + node.strideLen[0], node.count[1]);

        node.strideA[1] = new int[node.strideLen[0]];
        copy(node.strideA[0], node.strideA[0] + node.strideLen[0], node.strideA[1]);

        node.strideB[1] = new int[node.strideLen[0]];
        copy(node.strideB[0], node.strideB[0] + node.strideLen[0], node.strideB[1]);

        node.strideC[1] = new int[node.strideLen[0]];
        copy(node.strideC[0], node.strideC[0] + node.strideLen[0], node.strideC[1]);
    }

    static void matmul(Tensor<T>& A, Tensor<T>& B, Node<T>& node) {
        node.nEntries = 3;
        node.strideLen = new size_t[node.nEntries];
        node.reps = new int*[node.nEntries];
        node.count = new int*[node.nEntries];
        node.strideA = new int*[node.nEntries];
        node.strideB = new int*[node.nEntries];
        node.strideC = new int*[node.nEntries];

        tView<T> a = A.view();
        tView<T> b = B.view();
        tView<T> c = node.value.view();
        int* shapeBlock = new int[B.shapeLen * 2 + A.shapeLen * 2];
        tView<T> a_T = A.view();
        a_T.shape = shapeBlock;
        a_T.stride = a_T.shape + A.shapeLen;
        transp2D(A.shape, A.stride, A.shapeLen, a_T.shape, a_T.stride);
        tView<T> b_T = B.view();
        b_T.shape = a_T.stride + A.shapeLen;
        b_T.stride = b_T.shape + B.shapeLen;
        transp2D(B.shape, B.stride, B.shapeLen, b_T.shape, b_T.stride);

        _matmul(a, b, c, node.strideLen[0], node.reps[0], node.strideA[0], node.strideB[0], node.strideC[0]); 
        _matmul(c, b_T, a, node.strideLen[1], node.reps[1], node.strideC[1], node.strideB[1], node.strideA[1]); 
        _matmul(a_T, c, b, node.strideLen[2], node.reps[2], node.strideA[2], node.strideC[2], node.strideB[2]); 

        delete[] shapeBlock;
    }

    static void batch_matmul(Tensor<T>& A, Tensor<T>& B, Node<T>& node) {

        node.nEntries = 3;
        node.strideLen = new size_t[node.nEntries];
        node.reps = new int*[node.nEntries];
        node.count = new int*[node.nEntries];
        node.strideA = new int*[node.nEntries];
        node.strideB = new int*[node.nEntries];
        node.strideC = new int*[node.nEntries];
        
        tView<T> a = A.view();
        tView<T> b = B.view();
        tView<T> c = node.value.view();
        int* shapeBlock = new int[B.shapeLen * 2 + A.shapeLen * 2];
        tView<T> a_T = A.view();
        a_T.shape = shapeBlock;
        a_T.stride = a_T.shape + A.shapeLen;
        transp2D(A.shape, A.stride, A.shapeLen, a_T.shape, a_T.stride);
        tView<T> b_T = B.view();
        b_T.shape = a_T.stride + A.shapeLen;
        b_T.stride = b_T.shape + B.shapeLen;
        transp2D(B.shape, B.stride, B.shapeLen, b_T.shape, b_T.stride);

        _batch_matmul(a, b, c, node.strideLen[0], node.reps[0], node.count[0], node.strideA[0], node.strideB[0], node.strideC[0]); 
        _batch_matmul(c, b_T, a, node.strideLen[1], node.reps[1], node.count[1], node.strideC[1], node.strideB[1], node.strideA[1]); 
        _batch_matmul(a_T, c, b, node.strideLen[2], node.reps[2], node.count[2], node.strideA[2], node.strideC[2], node.strideB[2]); 
    }

    static void outer(Tensor<T>& A, Tensor<T>& B, Node<T>& node) {
        Tensor<T>& C = node.value;

        node.nEntries = 2;
        node.strideLen = new size_t[2];
        node.reps = new int*[2];
        node.count = new int*[2];
        node.strideA = new int*[2];
        node.strideB = new int*[2];
        node.strideC = new int*[2];

        _outer(A, B, C, node.strideLen[0], node.reps[0], node.count[0], node.strideA[0], node.strideB[0], node.strideC[0]);

        node.strideLen[1] = node.strideLen[0];

        node.reps[1] = new int[node.strideLen[0]];
        copy(node.reps[0], node.reps[0] + node.strideLen[0], node.reps[1]);

        node.count[1] = new int[node.strideLen[0]];
        copy(node.count[0], node.count[0] + node.strideLen[0], node.count[1]);

        node.strideA[1] = new int[node.strideLen[0]];
        copy(node.strideA[0], node.strideA[0] + node.strideLen[0], node.strideA[1]);

        node.strideB[1] = new int[node.strideLen[0]];
        copy(node.strideB[0], node.strideB[0] + node.strideLen[0], node.strideB[1]);

        node.strideC[1] = new int[node.strideLen[0]];
        copy(node.strideC[0], node.strideC[0] + node.strideLen[0], node.strideC[1]);
    }

    private:

    static void _flexible(Tensor<T>& A, Tensor<T>& B, Tensor<T>& C, size_t& strideLen, int*& reps, int*& count, int*& strideA, int*& strideB, int*& strideC) {
        strideLen = C.shapeLen;
        reps = new int[strideLen];
        copy(C.shape, C.shape + C.shapeLen, reps);
        for (int i = 0; i < strideLen; i++) {
            reps[i]--;
        }

        count = new int[strideLen];
        copy(reps, reps + strideLen, count);

        strideA = new int[strideLen];
        strideB = new int[strideLen];
        strideC = new int[strideLen];

        int idx, idxA, idxB, idxC;
        for (int i = 1; i <= strideLen; i++) {
            idx = strideLen - i;
            idxA = A.shapeLen - i;
            strideA[idx] = idxA >= 0 ? A.stride[idxA] : 0;
            idxB = B.shapeLen - i;
            strideB[idx] = idxB >= 0 ? B.stride[idxB] : 0;
            idxC = C.shapeLen - i;
            strideC[idx] = idxC >= 0 ? C.stride[idxC] : 0;
        }
    
        int offsetA = 0, _offsetA, offsetB = 0, _offsetB, offsetC = 0, _offsetC;
        for (int i = 1; i <= strideLen; i++) {
            idx = strideLen - i;

            idxA = A.shapeLen - i;
            _offsetA = offsetA;
            offsetA += ((idxA >= 0 ? A.shape[idxA] : i) - 1) * strideA[idx];
            strideA[idx] -= _offsetA;

            idxB = B.shapeLen - i;
            _offsetB = offsetB;
            offsetB += ((idxB >= 0 ? B.shape[idxB] : i) - 1) * strideB[idx];
            strideB[idx] -= _offsetB;

            idxC = C.shapeLen - i;
            _offsetC = offsetC;
            offsetC += ((idxC >= 0 ? C.shape[idxC] : i) - 1) * strideC[idx];
            strideC[idx] -= _offsetC;
        }
    }

    static void _matmul(tView<T>& A, tView<T>& B, tView<T>& C, size_t& strideLen, int*& reps, int*& strideA, int*& strideB, int*& strideC) {
        strideLen = 2;
        reps = new int[3] { C.shape[0], C.shape[1], A.shape[1] };

        strideA = new int[2];
        copy(A.stride, A.stride + A.shapeLen, strideA);
        strideB = new int[2];
        copy(B.stride, B.stride + B.shapeLen, strideB);
        strideC = new int[2];
        copy(C.stride, C.stride + C.shapeLen, strideC);

        int idx, idxC, offsetC = 0, _offsetC;
        for (int i = 1; i <= strideLen; i++) {
            idx = strideLen - i;
            idxC = C.shapeLen - i;
            _offsetC = offsetC;
            offsetC += ((idxC > 0 ? C.shape[idxC] : i) - 1) * strideC[idx];
            strideC[idx] -= _offsetC;
        }

        strideC[0] -= strideC[1];
    }
    
    static void _batch_matmul(tView<T>& A, tView<T>& B, tView<T>& C, size_t& strideLen, int*& reps, int*& count, int*& strideA, int*& strideB, int*& strideC) {
        strideLen = max(A.shapeLen, B.shapeLen);
        reps = new int[strideLen + 1];
        // save k into reps[strideLen]
        reps[strideLen] = A.shape[A.shapeLen - 1];

        combine_matrix(A.shape, A.shapeLen, B.shape, B.shapeLen, reps, strideLen);
        for (int i = 0; i < strideLen; i++) {
            reps[i]--;
        }

        count = new int[strideLen];
        copy(reps, reps + strideLen, count);

        strideA = new int[strideLen + 1];
        strideB = new int[strideLen + 1];
        strideC = new int[strideLen];

        strideA[strideLen] = A.stride[A.shapeLen - 1];
        strideB[strideLen] = B.stride[B.shapeLen - 2];

        int idx, idxA, idxB, idxC;
        for (int i = 1; i <= strideLen; i++) {
            idx = strideLen - i;
            idxA = A.shapeLen - i;
            strideA[idx] = idxA >= 0 ? A.stride[idxA] : 0;
            idxB = B.shapeLen - i;
            strideB[idx] = idxB >= 0 ? B.stride[idxB] : 0;
            idxC = C.shapeLen - i;
            strideC[idx] = idxC >= 0 ? C.stride[idxC] : 0;
        }


    
        strideA[strideLen - 1] = 0;
        strideB[strideLen - 2] = 0;
        int offsetA = 0, _offsetA, offsetB = 0, _offsetB, offsetC = 0, _offsetC;
        for (int i = 1; i <= strideLen; i++) {
            idx = strideLen - i;

            idxA = A.shapeLen - i;
            _offsetA = offsetA;
            offsetA += ((idxA >= 0 ? A.shape[idxA] : i) - 1) * strideA[idx];
            strideA[idx] -= _offsetA;

            idxB = B.shapeLen - i;
            _offsetB = offsetB;
            offsetB += ((idxB >= 0 ? B.shape[idxB] : i) - 1) * strideB[idx];
            strideB[idx] -= _offsetB;

            idxC = C.shapeLen - i;
            _offsetC = offsetC;
            offsetC += ((idxC >= 0 ? C.shape[idxC] : i) - 1) * strideC[idx];
            strideC[idx] -= _offsetC;
        }
    }

    static void _outer(Tensor<T>& A, Tensor<T>& B, Tensor<T>& C, size_t& strideLen, int*& reps, int*& count, int*& strideA, int*& strideB, int*& strideC) {
        strideLen = C.shapeLen;
        reps = new int[strideLen];
        copy(C.shape, C.shape + C.shapeLen, reps);
        for (int i = 0; i < strideLen; i++) {
            reps[i]--;
        }

        count = new int[strideLen];
        copy(reps, reps + strideLen, count);

        strideA = new int[strideLen];
        strideB = new int[strideLen];
        strideC = new int[strideLen];

        copy(C.stride, C.stride + C.shapeLen, strideC);
        copy(A.stride, A.stride + A.shapeLen, strideA);
        copy(B.stride, B.stride + B.shapeLen, strideB + A.shapeLen);

        // pad A.shape with 1s on the right
        int* a_shape_big = new int[strideLen];
        copy(A.shape, A.shape + A.shapeLen, a_shape_big);
        a_shape_big[A.shapeLen] = 1;

        int idx, idxA, idxB, idxC;
        int offsetA = 0, _offsetA, offsetB = 0, _offsetB, offsetC = 0, _offsetC;
        for (int i = 1; i <= strideLen; i++) {
            idx = strideLen - i;

            _offsetA = offsetA;
            offsetA += (a_shape_big[idx] - 1) * strideA[idx];
            strideA[idx] -= _offsetA;

            idxB = B.shapeLen - i;
            _offsetB = offsetB;
            offsetB += ((idxB >= 0 ? B.shape[idxB] : 1) - 1) * strideB[idx];
            strideB[idx] -= _offsetB;

            idxC = C.shapeLen - i;
            _offsetC = offsetC;
            offsetC += ((idxC >= 0 ? C.shape[idxC] : 1) - 1) * strideC[idx];
            strideC[idx] -= _offsetC;
        }
    }
};
