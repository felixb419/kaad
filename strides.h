#pragma once

#include "tensor.h"

template<typename T>
struct Strides {
    static void flexible(Tensor<T>* A, Tensor<T>* B, Tensor<T>* C, int*& strideA, int*& strideB, int*& strideC, size_t& strideLen) {
    
        strideLen = C->shapeLen;
        strideA = new int[strideLen];
        strideB = new int[strideLen];
        strideC = new int[strideLen];

        copy(C->stride, C->stride + C->shapeLen, strideC);

        int offsetA = C->shapeLen - A->shapeLen;
        int offsetB = C->shapeLen - B->shapeLen;
    
        for (size_t i = 0; i < C->shapeLen; i++) {
            int aDim = i - offsetA;
            strideA[i] = aDim >= 0 && A->shape[aDim] != 1 ? A->stride[aDim] : 0;
            int bDim = i - offsetB;
            strideB[i] = bDim >= 0 && B->shape[bDim] != 1 ? B->stride[bDim] : 0;
        }
        
        //int offset = 0, _offset, idx, idxA;
        //for (int i = 1; i <= strideLen; i++) {
        //    idx = strideLen - i;
        //    idxA = A->shapeLen - i;
        //    _offset = offset;
        //    offset += ((idxA > 0 ? A->shape[idxA] : 1) - 1) * strideA[idx];
        //    strideA[idx] -= _offset;
        //}
    }
};