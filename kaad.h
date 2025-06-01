#pragma once

#include "operations.h"
#include "tensor.h"
#include "gradients.h"
#include "recorder.h"
#include "strides.h"

using namespace std;

// add A and B
// where A and B are Tensors with Broadcastable shapes
template <typename T>
int add(Recorder<T>& rec, int indA, int indB) {
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

        rec.nodes.emplace_back(Operations<T>::scalarAdd, Gradients<T>::scalarAdd_grad, tensor, scalar, newShape, tensor_ref.shapeLen);
        rec.nodes[recLen].strideLen[0] = tensor_ref.shapeLen;
    }
    else if (A.shapeLen == B.shapeLen && equal(A.shape, A.shape + A.shapeLen, B.shape)) {
        int* newShape = new int[A.shapeLen];
        copy(A.shape, A.shape + A.shapeLen, newShape);

        rec.nodes.emplace_back(Operations<T>::pointAdd, Gradients<T>::pointAdd_grad, indA, indB, newShape, A.shapeLen);
        rec.nodes[recLen].strideLen[0] = A.shapeLen;
    }
    else {
        size_t newLen = max(A.shapeLen, B.shapeLen);
        int* newShape = new int[newLen];
        combine_flexible(A.shape, A.shapeLen, B.shape, B.shapeLen, newShape, newLen);

        rec.nodes.emplace_back(Operations<T>::flexAdd, Gradients<T>::flexAdd_grad, indA, indB, newShape, newLen);
        Strides<T>::flexible(&A, &B, rec.nodes[recLen]); 
    }
    return recLen;
}

template <typename T>
int sub(Recorder<T>& rec, int indA, int indB) {
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
        
        rec.nodes.emplace_back(Operations<T>::scalarSub, Gradients<T>::scalarSub_grad, tensor, scalar, newShape, tensor_ref.shapeLen);
        rec.nodes[recLen].strideLen[0] = tensor_ref.shapeLen;
    }
    else if (A.shapeLen == B.shapeLen && equal(A.shape, A.shape + A.shapeLen, B.shape)) {
        int* newShape = new int[A.shapeLen];
        copy(A.shape, A.shape + A.shapeLen, newShape);

        rec.nodes.emplace_back(Operations<T>::pointSub, Gradients<T>::pointSub_grad, indA, indB, newShape, A.shapeLen);
        rec.nodes[recLen].strideLen[0] = A.shapeLen;
    }
    else {
        size_t newLen = max(A.shapeLen, B.shapeLen);
        int* newShape = new int[newLen];
        combine_flexible(A.shape, A.shapeLen, B.shape, B.shapeLen, newShape, newLen);

        rec.nodes.emplace_back(Operations<T>::flexSub, Gradients<T>::flexSub_grad, indA, indB, newShape, newLen);
        Strides<T>::flexible(&A, &B, rec.nodes[recLen]); 
    }
    return recLen;
}

template <typename T>
int mul(Recorder<T>& rec, int indA, int indB) {
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

        rec.nodes.emplace_back(Operations<T>::scalarMul, Gradients<T>::scalarMul_grad, tensor, scalar, newShape, tensor_ref.shapeLen);
        rec.nodes[recLen].strideLen[0] = tensor_ref.shapeLen;
    }
    else if (A.shapeLen == B.shapeLen && equal(A.shape, A.shape + A.shapeLen, B.shape)) {
        int* newShape = new int[A.shapeLen];
        copy(A.shape, A.shape + A.shapeLen, newShape);
        
        rec.nodes.emplace_back(Operations<T>::pointMul, Gradients<T>::pointMul_grad, indA, indB, newShape, A.shapeLen);
        Node<T>& node = rec.nodes[recLen];
        Strides<T>::flexible(&A, &B, rec.nodes[recLen]); 
    }
    else {
        size_t newLen = max(A.shapeLen, B.shapeLen);
        int* newShape = new int[newLen];
        combine_flexible(A.shape, A.shapeLen, B.shape, B.shapeLen, newShape, newLen);

        rec.nodes.emplace_back(Operations<T>::flexMul, Gradients<T>::flexMul_grad, indA, indB, newShape, newLen);
        Strides<T>::flexible(&A, &B, rec.nodes[recLen]); 
    }
    return recLen;
}

template <typename T>
int div(Recorder<T>& rec, int indA, int indB) {
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
        
        rec.nodes.emplace_back(Operations<T>::scalarDiv, Gradients<T>::scalarDiv_grad, tensor, scalar, newShape, tensor_ref.shapeLen);
        rec.nodes[recLen].strideLen[0] = tensor_ref.shapeLen;
    }
    else if (A.shapeLen == B.shapeLen && equal(A.shape, A.shape + A.shapeLen, B.shape)) {
        int* newShape = new int[A.shapeLen];
        copy(A.shape, A.shape + A.shapeLen, newShape);
        
        rec.nodes.emplace_back(Operations<T>::pointDiv, Gradients<T>::pointDiv_grad, indA, indB, newShape, A.shapeLen);
        rec.nodes[recLen].strideLen[0] = A.shapeLen;
    }
    else {
        size_t newLen = max(A.shapeLen, B.shapeLen);
        int* newShape = new int[newLen];
        combine_flexible(A.shape, A.shapeLen,
            B.shape, B.shapeLen,
            newShape, newLen);

        rec.nodes.emplace_back(Operations<T>::flexDiv, Gradients<T>::flexDiv_grad, indA, indB, newShape, newLen);
        Strides<T>::flexible(&A, &B, rec.nodes[recLen]); 
    }
    return recLen;
}

template <typename T>
int pow(Recorder<T>& rec, int indA, int indB) {
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
        
        rec.nodes.emplace_back(Operations<T>::scalarPow, Gradients<T>::scalarPow_grad, recLen, tensor, scalar, newShape, tensor_ref.shapeLen);
    }
    else if (A.shapeLen == B.shapeLen && equal(A.shape, A.shape + A.shapeLen, B.shape)) {
        int* newShape = new int[A.shapeLen];
        copy(A.shape, A.shape + A.shapeLen, newShape);
        
        rec.nodes.emplace_back(Operations<T>::pointPow, Gradients<T>::pointPow_grad, indA, indB, newShape, A.shapeLen);
    }
    else {
        size_t newLen = max(A.shapeLen, B.shapeLen);
        int* newShape = new int[newLen];
        combine_flexible(A.shape, A.shapeLen, B.shape, B.shapeLen, newShape, newLen);

        rec.nodes.emplace_back(Operations<T>::flexPow, Gradients<T>::flexPow_grad, indA, indB, newShape, newLen);
    }
    return recLen;
}

template <typename T>
int negative(Recorder<T>& rec, int indA) {
    int recLen = rec.nodes.size();
    Tensor<T>& A = rec.nodes[indA].value;

    int* newShape = new int[A.shapeLen];
    copy(A.shape, A.shape + A.shapeLen, newShape);

    rec.nodes.emplace_back(Operations<T>::negate, Gradients<T>::negate_grad, indA, -1, newShape, A.shapeLen);
    return recLen;
}

template <typename T>
int square(Recorder<T>& rec, int indA) {
    int recLen = rec.nodes.size();
    Tensor<T>& A = rec.nodes[indA].value;

    int* newShape = new int[A.shapeLen];
    copy(A.shape, A.shape + A.shapeLen, newShape);

    rec.nodes.emplace_back(Operations<T>::square, Gradients<T>::square_grad, indA, -1, newShape, A.shapeLen);
    return recLen;
}

template <typename T>
int sqrt(Recorder<T>& rec, int indA) {
    int recLen = rec.nodes.size();
    Tensor<T>& A = rec.nodes[indA].value;
    int* newShape = new int[A.shapeLen];
    copy(A.shape, A.shape + A.shapeLen, newShape);

    rec.nodes.emplace_back(Operations<T>::sqrt, Gradients<T>::sqrt_grad, indA, -1, newShape, A.shapeLen);
    return recLen;
}

template <typename T>
int log(Recorder<T>& rec, int indA) {
    int recLen = rec.nodes.size();
    Tensor<T>& A = rec.nodes[indA].value;
    int* newShape = new int[A.shapeLen];
    copy(A.shape, A.shape + A.shapeLen, newShape);

    rec.nodes.emplace_back(Operations<T>::log, Gradients<T>::log_grad, indA, -1, newShape, A.shapeLen);
    return recLen;
}

template <typename T>
int exp(Recorder<T>& rec, int indA) {
    int recLen = rec.nodes.size();
    Tensor<T>& A = rec.nodes[indA].value;
    int* newShape = new int[A.shapeLen];
    copy(A.shape, A.shape + A.shapeLen, newShape);

    rec.nodes.emplace_back(Operations<T>::exp, Gradients<T>::exp_grad, indA, -1, newShape, A.shapeLen);
    return recLen;
}

template <typename T>
int abs(Recorder<T>& rec, int indA) {
    int recLen = rec.nodes.size();
    Tensor<T>& A = rec.nodes[indA].value;
    int* newShape = new int[A.shapeLen];
    copy(A.shape, A.shape + A.shapeLen, newShape);

    rec.nodes.emplace_back(Operations<T>::abs, Gradients<T>::abs_grad, indA, -1, newShape, A.shapeLen);
    return recLen;
}

template <typename T>
int dot(Recorder<T>& rec, int indA, int indB) {
    int recLen = rec.nodes.size();
    Tensor<T>& A = rec.nodes[indA].value;
    Tensor<T>& B = rec.nodes[indB].value;

    bool A_scalar = A.shapeLen == 0 && A.shape[0] == 1;
    bool B_scalar = B.shapeLen == 0 && B.shape[0] == 1;

    if (B_scalar) {
        rec.nodes.emplace_back(Operations<T>::scalarDot, Gradients<T>::scalarDot_grad, indA, indB, ((T)0));
    }
    else if (A_scalar) {
        rec.nodes.emplace_back(Operations<T>::scalarDot, Gradients<T>::scalarDot_grad, indB, indA, ((T)0));
    }
    else if (A.shapeLen == B.shapeLen && equal(A.shape, A.shape + A.shapeLen, B.shape)) {
        rec.nodes.emplace_back(Operations<T>::dot, Gradients<T>::dot_grad, indA, indB, ((T)0));
    }
    else {
        throw invalid_argument("shape error");
    }

    return recLen;
}

template <typename T>
int outer(Recorder<T>& rec, int indA, int indB) {
    int recLen = rec.nodes.size();
    Tensor<T>& A = rec.nodes[indA].value;
    Tensor<T>& B = rec.nodes[indB].value;

    int offsetA = max(((int)A.shapeLen) - ((int)B.shapeLen), 0);
    int offsetB = max(((int)B.shapeLen) - ((int)A.shapeLen), 0);

    Tensor<T>& small = A.shapeLen < B.shapeLen ? A : B;
    for (int i = 0; i < small.shapeLen; i++) {
        if (A.shape[i + offsetA] != B.shape[i + offsetB]) {
            throw invalid_argument("shape error");
        }
    }

    size_t newLen = A.shapeLen + B.shapeLen;
    int* newShape = new int[newLen];
    copy(A.shape, A.shape + A.shapeLen, newShape);
    copy(B.shape, B.shape + B.shapeLen, newShape + A.shapeLen);

    rec.nodes.emplace_back(Operations<T>::outer, Gradients<T>::outer_grad, indA, indB, newShape, newLen);

    return recLen;
}

template <typename T>
int matmul(Recorder<T>& rec, int indA, int indB) {
    int recLen = rec.nodes.size();
    Tensor<T>& A = rec.nodes[indA].value;
    Tensor<T>& B = rec.nodes[indB].value;

    size_t newLen = max(A.shapeLen, B.shapeLen);
    int* newShape = new int[newLen];
    combine_matrix(A.shape, A.shapeLen, B.shape, B.shapeLen, newShape, newLen);

    if (newLen == 2) {
        rec.nodes.emplace_back(Operations<T>::matmul, Gradients<T>::matmul_grad, indA, indB, newShape, newLen);
    }
    else {
        rec.nodes.emplace_back(Operations<T>::batch_matmul, Gradients<T>::batch_matmul_grad, indA, indB, newShape, newLen);
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
int sum(Recorder<T>& rec, int indA) {
    int recLen = rec.nodes.size();
    
    int newShape[] = {1};
    rec.nodes.emplace_back(Operations<T>::sum, Gradients<T>::sum_grad, indA, -1, newShape, 1);

    return recLen;
}

template <typename T>
int sum(Recorder<T>& rec, int indA, int dim) {
    int recLen = rec.nodes.size();
    Tensor<T>& A = rec.nodes[indA].value;

    size_t newLen = A.shapeLen - 1;
    int* newShape = new int[newLen];
    copy(A.shape, A.shape + dim, newShape);
    for (int i = dim; i < newLen; i++) {
        newShape[i] = A.shape[i + 1];
    }

    rec.nodes.emplace_back(Operations<T>::sum_dim, Gradients<T>::sum_dim_grad, indA, -1, newShape, newLen);
    // save dim into ctx
    rec.nodes[recLen].ctx = new int[] { dim };
    rec.nodes[recLen].ctx_is_array = true;

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
int transpose(Recorder<T>& rec, int indA) {
    int recLen = rec.nodes.size();
    Tensor<T>& A = rec.nodes[indA].value;

    if (A.shapeLen < 2) {
        throw invalid_argument("A has to have more than 1 dimension to transpose it");
    }

    int* shape_T = new int[A.shapeLen];
    int* stride_T = new int[A.shapeLen];
    copy(A.shape, A.shape + A.shapeLen, shape_T);
    copy(A.stride, A.stride + A.shapeLen, stride_T);
    
    transp(A.shape, A.stride, shape_T, stride_T, A.shapeLen);

    rec.nodes.emplace_back(Operations<T>::transpose, Gradients<T>::transp_grad, indA, -1, shape_T, stride_T, A.shapeLen);

    return recLen;
}

template <typename T>
int transpose(Recorder<T>& rec, int indA, initializer_list<int> permutation) {
    int recLen = rec.nodes.size();
    Tensor<T>& A = rec.nodes[indA].value;

    if (A.shapeLen < 2) {
        throw invalid_argument("A has to have more than 1 dimension to transpose it");
    }
    if (permutation.size() != A.shapeLen) {
        throw invalid_argument("permutation has to have same size as shapeLen of A");
    }

    int* shape_T = new int[A.shapeLen];
    int* stride_T = new int[A.shapeLen];

    int* sh = shape_T;
    int* st = stride_T;
    for (int idx : permutation) {
        *(sh++) = A.shape[idx];
        *(st++) = A.stride[idx];
    } 

    rec.nodes.emplace_back(Operations<T>::transpose, Gradients<T>::transp_grad, indA, -1, shape_T, stride_T, A.shapeLen);

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
