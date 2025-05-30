#pragma once

#include "tensor.h"

template<typename T>
struct Strides {
    static void flexible(Tensor<T>* A, Tensor<T>* B, Tensor<T>* C, int*& strideA, int*& strideB, int*& strideC, int*& reps, int*& count, size_t& strideLen) {

        strideLen = C->shapeLen;
        reps = new int[strideLen];
        strideA = new int[strideLen];
        strideB = new int[strideLen];
        strideC = new int[strideLen];

        copy(C->shape, C->shape + C->shapeLen, reps);
        for (int i = 0; i < strideLen; i++) {
            reps[i]--;
        }
        count = new int[strideLen];
        copy(reps, reps + strideLen, count);

        copy(C->stride, C->stride + C->shapeLen, strideC);

        int idx, idxA, idxB, idxC;
        for (int i = 1; i <= strideLen; i++) {
            idx = strideLen - i;
            idxA = A->shapeLen - i;
            strideA[idx] = idxA >= 0 ? A->stride[idxA] : 0;
            idxB = B->shapeLen - i;
            strideB[idx] = idxB >= 0 ? B->stride[idxB] : 0;
            idxC = C->shapeLen - i;
            strideC[idx] = idxC >= 0 ? C->stride[idxC] : 0;
        }
    
        int offsetA = 0, _offsetA, offsetB = 0, _offsetB, offsetC = 0, _offsetC;
        for (int i = 1; i <= strideLen; i++) {
            idx = strideLen - i;

            idxA = A->shapeLen - i;
            _offsetA = offsetA;
            offsetA += ((idxA > 0 ? A->shape[idxA] : i) - 1) * strideA[idx];
            strideA[idx] -= _offsetA;

            idxB = B->shapeLen - i;
            _offsetB = offsetB;
            offsetB += ((idxB > 0 ? B->shape[idxB] : i) - 1) * strideB[idx];
            strideB[idx] -= _offsetB;

            idxC = C->shapeLen - i;
            _offsetC = offsetC;
            offsetC += ((idxC > 0 ? C->shape[idxC] : i) - 1) * strideC[idx];
            strideC[idx] -= _offsetC;
        }

        //for (int i = 0; i < strideLen - 1; i++) {
        //    strideA[i] -= strideA[i+1];
        //    strideB[i] -= strideB[i+1];
        //    strideC[i] -= strideC[i+1];
        //}

    }
};
