#pragma once

#include "operations.h"
#include "tensor.h"
#include "gradients.h"
#include "recorder.h"
#include "strides.h"
#include "unaryOps.h"
#include "binaryOps.h"

#include <iostream>
#include <math.h>

using namespace std;

template <typename T>
int tensordot(Recorder<T>& rec, int indA, int indB, int dims) {
    int recLen = rec.nodes.size();
    Tensor<T>& A = rec.nodes[indA].value;
    Tensor<T>& B = rec.nodes[indB].value;

    if (dims > 0 && !(A.shapeLen == B.shapeLen && equal(A.shape, A.shape + A.shapeLen, B.shape))) {
        throw invalid_argument("shape error");
    }
        
    int offsetA = max(((int)A.shapeLen) - ((int)B.shapeLen), 0);
    int offsetB = max(((int)B.shapeLen) - ((int)A.shapeLen), 0);

    Tensor<T>& small = A.shapeLen < B.shapeLen ? A : B;
    for (int i = 0; i < small.shapeLen; i++) {
        if (A.shape[i + offsetA] != B.shape[i + offsetB]) {
            throw invalid_argument("shape error");
        }
    }

    // assemble c_big
    size_t newLen = A.shapeLen + B.shapeLen;
    int* newShape = new int[newLen];
    copy(A.shape, A.shape + A.shapeLen, newShape);
    copy(B.shape, B.shape + B.shapeLen, newShape + A.shapeLen);

    if (dims == 0) {
        rec.nodes.emplace_back(Operations<T>::outer, Gradients<T>::outer_grad, indA, indB, newShape, newLen);
    }
    else {
        // reduce newShape
        size_t newLen_small = newLen - dims * 2;
        int* newShape_small = new int[newLen_small];
        copy(A.shape, A.shape + A.shapeLen - dims, newShape_small);
        copy(B.shape + dims, B.shape + B.shapeLen, newShape_small + A.shapeLen - dims);
        // append node with tensordot
        rec.nodes.emplace_back(Operations<T>::tensordot, Gradients<T>::tensordot_grad, indA, indB, newShape_small, newLen_small);
        // make ctx, dims followed with newShape
        int* context = new int[newLen + 2];
        context[0] = dims;
        copy(newShape, newShape + newLen, context + 1);
        rec.nodes[recLen].ctx = context;
    }

    return recLen;
}

template <typename T>
int minimum(Recorder<T>& rec, int indA, int indB) {
    int recLen = rec.nodes.size();
    Tensor<T>& A = rec.nodes[indA].value;
    Tensor<T>& B = rec.nodes[indB].value;
    bool A_scalar = A.shapeLen == 0 && A.shape[0] == 1;
    bool B_scalar = B.shapeLen == 0 && B.shape[0] == 1;

    if (A_scalar || B_scalar) {
        int tensor = A_scalar ? indB : indA;
        int scalar = A_scalar ? indA : indB;
        Tensor<T>& tensor_ref = rec.nodes[tensor].value;
        int* newShape = new int[tensor_ref.shapeLen];
        copy(tensor_ref.shape, tensor_ref.shape + tensor_ref.shapeLen, newShape);

        rec.nodes.emplace_back(Operations<T>::scalarMin, Gradients<T>::scalarMin_grad, tensor, scalar, newShape, tensor_ref.shapeLen);
    }
    else if (A.shapeLen == B.shapeLen && equal(A.shape, A.shape + A.shapeLen, B.shape)) {
        int* newShape = new int[A.shapeLen];
        copy(A.shape, A.shape + A.shapeLen, newShape);

        rec.nodes.emplace_back(Operations<T>::pointMin, Gradients<T>::pointMin_grad, indA, indB, newShape, A.shapeLen);
    }
    else {
        size_t newLen = max(A.shapeLen, B.shapeLen);
        int* newShape = new int[newLen];
        combine_flexible(A.shape, A.shapeLen, B.shape, B.shapeLen, newShape, newLen);

        rec.nodes.emplace_back(Operations<T>::flexMin, Gradients<T>::flexMin_grad, indA, indB, newShape, newLen);
    }
    return recLen;
}

template<typename T>
int maximum(Recorder<T>& rec, int indA, int indB) {
    int recLen = rec.nodes.size();
    Tensor<T>& A = rec.nodes[indA].value;
    Tensor<T>& B = rec.nodes[indB].value;
    bool A_scalar = A.shapeLen == 0 && A.shape[0] == 1;
    bool B_scalar = B.shapeLen == 0 && B.shape[0] == 1;

    if (A_scalar || B_scalar) {
        int tensor = A_scalar ? indB : indA;
        int scalar = A_scalar ? indA : indB;
        Tensor<T>& tensor_ref = rec.nodes[tensor].value;
        int* newShape = new int[tensor_ref.shapeLen];
        copy(tensor_ref.shape, tensor_ref.shape + tensor_ref.shapeLen, newShape);

        rec.nodes.emplace_back(Operations<T>::scalarMax, Gradients<T>::scalarMax_grad, tensor, scalar, newShape, tensor_ref.shapeLen);
    }
    else if (A.shapeLen == B.shapeLen && equal(A.shape, A.shape + A.shapeLen, B.shape)) {
        int* newShape = new int[A.shapeLen];
        copy(A.shape, A.shape + A.shapeLen, newShape);

        rec.nodes.emplace_back(Operations<T>::pointMax, Gradients<T>::pointMax_grad, indA, indB, newShape, A.shapeLen);
    }
    else {
        size_t newLen = max(A.shapeLen, B.shapeLen);
        int* newShape = new int[newLen];
        combine_flexible(A.shape, A.shapeLen, B.shape, B.shapeLen, newShape, newLen);

        rec.nodes.emplace_back(Operations<T>::flexMax, Gradients<T>::flexMax_grad, indA, indB, newShape, newLen);
    }
    return recLen;
}

template <typename T>
int mean(Recorder<T>& rec, int indA) {
    int recLen = rec.nodes.size();
    Tensor<T>& A = rec.nodes[indA].value;
    
    int newShape[] = {1};
    rec.nodes.emplace_back(Operations<T>::mean, Gradients<T>::mean_grad, indA, -1, newShape, 1);

    return recLen;
}

template <typename T>
int mean(Recorder<T>& rec, int indA, int dim) {
    int recLen = rec.nodes.size();
    Tensor<T>& A = rec.nodes[indA].value;

    size_t newLen = A.shapeLen - 1;
    int* newShape = new int[newLen];
    copy(A.shape, A.shape + dim, newShape);
    for (int i = dim; i < newLen; i++) {
        newShape[i] = A.shape[i + 1];
    }
    
    rec.nodes.emplace_back(Operations<T>::mean_dim, Gradients<T>::mean_dim_grad, indA, -1, newShape, newLen);
    rec.nodes[recLen].ctx = new int[] { dim };
    rec.nodes[recLen].ctx_is_array = true;

    return recLen;
}


template <typename T>
int tile(Recorder<T>& rec, int indA, initializer_list<int> multiples) {
    int recLen = rec.nodes.size();
    Tensor<T>& A = rec.nodes[indA].value;

    size_t m_len = multiples.size();
    int* m = new int[m_len];
    copy(multiples.begin(), multiples.end(), m);

    int* newShape;
    size_t newLen;


    if (A.shapeLen >= m_len) {
        newLen = A.shapeLen;
        newShape = new int[newLen];
        copy(A.shape, A.shape + newLen, newShape);

        int offset = A.shapeLen - m_len;
        int* s = newShape;
        for (int m : multiples) {
            *(s++ + offset) *= m;
        }
    }
    else {
        newLen = A.shapeLen;
        newShape = new int[newLen];        
        copy(m, m + m_len, newShape);

        int offset = m_len - A.shapeLen;
        for (int i = 0; i < A.shapeLen; i++) {
            newShape[i + offset] *= A.shape[i];
        }

    }

    // save multiples into node
    Tensor<T> temp;
    temp.shape = m;
    rec.nodes.emplace_back(move(temp));

    rec.nodes.emplace_back(Operations<T>::tile, Gradients<T>::tile_grad, indA, recLen, newShape, newLen);

    return recLen + 1;
}

template <typename T>
int slice(Recorder<T>& rec, int indA, initializer_list<int> start, initializer_list<int> length) {
    int recLen = rec.nodes.length();
    Tensor<T>& A = rec.nodes[indA].value;

    size_t newLen = length.size();
    int* newShape = new int[newLen];
    copy(length.begin(), length.end(), newShape);
    int* offset = new int[newLen];
    copy(start.begin(), start.end(), offset);

    rec.nodes.emplace_back(Operations<T>::slice, Gradients<T>::slice_grad, indA, -1, newShape, newLen);
    // save offset into ctx
    rec.nodes[recLen].ctx = offset;
    rec.nodes[recLen].ctx_is_array = true;

    return recLen;
}
