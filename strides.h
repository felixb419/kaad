#pragma once

#include "tensor.h"
#include "recorder.h"

template<typename T>
struct Strides {
    static void flexible(Tensor<T>* A, Tensor<T>* B, Node<T>& node) {
        Tensor<T>* C = &node.value;

        node.nEntries = 1;
        node.strideLen = new size_t[1];
        size_t& strideLen = node.strideLen[0];
        node.reps = new int*[1];
        int*& reps = node.reps[0];
        node.count = new int*[1];
        int*& count = node.count[0]; 
        node.strideA = new int*[1];
        int*& strideA = node.strideA[0];
        node.strideB = new int*[1];
        int*& strideB = node.strideB[0];
        node.strideC = new int*[1];
        int*& strideC = node.strideC[0];

        strideLen = C->shapeLen;

        reps = new int[strideLen];
        copy(C->shape, C->shape + C->shapeLen, reps);
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
    }
};
