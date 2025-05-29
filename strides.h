#pragma once

#include "tensor.h"

template<typename T>
struct Strides {
    static void flexible(Tensor<T>* A, Tensor<T>* B, Tensor<T>* C, int*& strideA, int*& strideB, int*& strideC, int*& reps, size_t& strideLen) {

        strideLen = C->shapeLen;
        reps = new int[strideLen];
        strideA = new int[strideLen];
        strideB = new int[strideLen];
        strideC = new int[strideLen];

        copy(C->shape, C->shape + C->shapeLen, reps);
        for (int i = 0; i < strideLen; i++) {
            reps[i]--;
        }

        copy(C->stride, C->stride + C->shapeLen, strideC);

        int diffA = C->shapeLen - A->shapeLen;
        int diffB = C->shapeLen - B->shapeLen;
    
        for (size_t i = 0; i < C->shapeLen; i++) {
            int aDim = i - diffA;
            strideA[i] = aDim >= 0 && A->shape[aDim] != 1 ? A->stride[aDim] : 0;
            int bDim = i - diffB;
            strideB[i] = bDim >= 0 && B->shape[bDim] != 1 ? B->stride[bDim] : 0;
        }
        
        int offsetA = 0, _offsetA, offsetB = 0, _offsetB, offsetC = 0, _offsetC;
        int idx, idxA, idxB, idxC;
        for (int i = 1; i <= strideLen; i++) {
            idx = strideLen - i;

            idxA = A->shapeLen - i;
            _offsetA = offsetA;
            offsetA += ((idxA > 0 ? A->shape[idxA] : 1) - 1) * strideA[idx];
            strideA[idx] -= _offsetA;

            idxB = B->shapeLen - i;
            _offsetB = offsetB;
            offsetB += ((idxB > 0 ? B->shape[idxB] : 1) - 1) * strideB[idx];
            strideB[idx] -= _offsetB;

            idxC = C->shapeLen - i;
            _offsetC = offsetC;
            offsetC += ((idxC > 0 ? C->shape[idxC] : 1) - 1) * strideC[idx];
            strideC[idx] -= _offsetC;
        }

        for (int i = 0; i < strideLen - 1; i++) {
            strideA[i] -= strideA[i + 1];
            strideB[i] -= strideB[i + 1];
            strideC[i] -= strideC[i + 1];
        }
    }
};
