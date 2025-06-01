#pragma once

#include "tensor.h"
#include "recorder.h"

template<typename T>
struct Strides {
    static void flexible(Tensor<T>* A, Tensor<T>* B, Node<T>& node) {
        Tensor<T>* C = &node.value;

        node.strideLen[0] = C->shapeLen;
        node.reps[0] = new int[node.strideLen[0]];
        node.strideA[0] = new int[node.strideLen[0]];
        node.strideB[0] = new int[node.strideLen[0]];
        node.strideC[0] = new int[node.strideLen[0]];

        copy(C->shape, C->shape + C->shapeLen, node.reps[0]);
        for (int i = 0; i < node.strideLen[0]; i++) {
            node.reps[i]--;
        }
        node.count[0] = new int[node.strideLen[0]];
        copy(node.reps[0], node.reps[0] + node.strideLen[0], node.count[0]);

        copy(C->stride, C->stride + C->shapeLen, node.strideC[0]);

        int idx, idxA, idxB, idxC;
        for (int i = 1; i <= node.strideLen[0]; i++) {
            idx = node.strideLen[0] - i;
            idxA = A->shapeLen - i;
            node.strideA[0][idx] = idxA >= 0 ? A->stride[idxA] : 0;
            idxB = B->shapeLen - i;
            node.strideB[0][idx] = idxB >= 0 ? B->stride[idxB] : 0;
            idxC = C->shapeLen - i;
            node.strideC[0][idx] = idxC >= 0 ? C->stride[idxC] : 0;
        }
    
        int offsetA = 0, _offsetA, offsetB = 0, _offsetB, offsetC = 0, _offsetC;
        for (int i = 1; i <= node.strideLen[0]; i++) {
            idx = node.strideLen[0] - i;

            idxA = A->shapeLen - i;
            _offsetA = offsetA;
            offsetA += ((idxA > 0 ? A->shape[idxA] : i) - 1) * node.strideA[0][idx];
            node.strideA[0][idx] -= _offsetA;

            idxB = B->shapeLen - i;
            _offsetB = offsetB;
            offsetB += ((idxB > 0 ? B->shape[idxB] : i) - 1) * node.strideB[0][idx];
            node.strideB[0][idx] -= _offsetB;

            idxC = C->shapeLen - i;
            _offsetC = offsetC;
            offsetC += ((idxC > 0 ? C->shape[idxC] : i) - 1) * node.strideC[0][idx];
            node.strideC[0][idx] -= _offsetC;
        }
    }
};
