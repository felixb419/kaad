#pragma once

#include "operations.h"
#include "tensor.h"
#include "gradients.h"
#include "recorder.h"

#include <iostream>

using namespace std;

template <typename T>
int add(Recorder<T>& rec, int indA, int indB) {
    int recLen = rec.nodes.size();
    bool A_scalar = rec.nodes[indA].value.shapeLen == 0 && rec.nodes[indA].value.shape[0] == 1;
    bool B_scalar = rec.nodes[indB].value.shapeLen == 0 && rec.nodes[indB].value.shape[0] == 1;

    if (A_scalar || B_scalar) {
        int tensor = A_scalar ? indB : indA;
        int scalar = A_scalar ? indA : indB;
        Node<T> temp(Operations<T>::scalarAdd, Gradients<T>::scalarAdd_grad,
            recLen, tensor, scalar, rec.nodes[tensor].value.shape, rec.nodes[tensor].value.shapeLen);
        rec.nodes.emplace_back(temp);
        //rec.nodes.emplace_back(scalarAdd, scalarAdd_grad,
        //    recLen, tensor, scalar, rec.nodes[tensor].value.shape, rec.nodes[tensor].value.shapeLen);
    }
    else if (rec.nodes[indA].value.shapeLen == rec.nodes[indB].value.shapeLen &&
            equal(rec.nodes[indA].value.shape, rec.nodes[indA].value.shape + rec.nodes[indA].value.shapeLen,
            rec.nodes[indB].value.shape)) {
        
        rec.nodes.emplace_back(Operations<T>::pointAdd, Gradients<T>::pointAdd_grad,
            recLen, indA, indB, rec.nodes[indA].value.shape, rec.nodes[indA].value.shapeLen);
    }
    else {
        size_t newLen = max(rec.nodes[indA].value.shapeLen, rec.nodes[indB].value.shapeLen);
        int* newShape = new int[newLen];
        combine_flexible(rec.nodes[indA].value.shape, rec.nodes[indA].value.shapeLen,
            rec.nodes[indB].value.shape, rec.nodes[indB].value.shapeLen,
            newShape, newLen);

        rec.nodes.emplace_back(Operations<T>::flexAdd, Gradients<T>::flexAdd_grad,
            recLen, indA, indB, newShape, newLen);
    }
    return recLen;
}

template <typename T>
int sub(Recorder<T>& rec, int indA, int indB) {
    int recLen = rec.nodes.size();
    bool A_scalar = rec.nodes[indA].value.shapeLen == 0 && rec.nodes[indA].value.shape[0] == 1;
    bool B_scalar = rec.nodes[indB].value.shapeLen == 0 && rec.nodes[indB].value.shape[0] == 1;

    if (A_scalar || B_scalar) {
        int tensor = A_scalar ? indB : indA;
        int scalar = A_scalar ? indA : indB;
        
        rec.nodes.emplace_back(Operations<T>::scalarSub, Gradients<T>::scalarSub_grad,
            recLen, tensor, scalar, rec.nodes[tensor].value.shape, rec.nodes[tensor].value.shapeLen);
    }
    else if (rec.nodes[indA].value.shapeLen == rec.nodes[indB].value.shapeLen &&
            equal(rec.nodes[indA].value.shape, rec.nodes[indA].value.shape + rec.nodes[indA].value.shapeLen,
            rec.nodes[indB].value.shape)) {
        
        rec.nodes.emplace_back(Operations<T>::pointSub, Gradients<T>::pointSub_grad,
            recLen, indA, indB, rec.nodes[indA].value.shape, rec.nodes[indA].value.shapeLen);
    }
    else {
        size_t newLen = max(rec.nodes[indA].value.shapeLen, rec.nodes[indB].value.shapeLen);
        int* newShape = new int[newLen];
        combine_flexible(rec.nodes[indA].value.shape, rec.nodes[indA].value.shapeLen,
            rec.nodes[indB].value.shape, rec.nodes[indB].value.shapeLen,
            newShape, newLen);

        rec.nodes.emplace_back(Operations<T>::flexSub, Gradients<T>::flexSub_grad,
            recLen, indA, indB, newShape, newLen);
    }
    return recLen;
}

template <typename T>
int mul(Recorder<T>& rec, int indA, int indB) {
    int recLen = rec.nodes.size();
    bool A_scalar = rec.nodes[indA].value.shapeLen == 0 && rec.nodes[indA].value.shape[0] == 1;
    bool B_scalar = rec.nodes[indB].value.shapeLen == 0 && rec.nodes[indB].value.shape[0] == 1;

    if (A_scalar || B_scalar) {
        int tensor = A_scalar ? indB : indA;
        int scalar = A_scalar ? indA : indB;
        
        rec.nodes.emplace_back(Operations<T>::scalarMul, Gradients<T>::scalarMul_grad,
            recLen, tensor, scalar, rec.nodes[tensor].value.shape, rec.nodes[tensor].value.shapeLen);
    }
    else if (rec.nodes[indA].value.shapeLen == rec.nodes[indB].value.shapeLen &&
            equal(rec.nodes[indA].value.shape, rec.nodes[indA].value.shape + rec.nodes[indA].value.shapeLen,
            rec.nodes[indB].value.shape)) {
        
        rec.nodes.emplace_back(Operations<T>::pointMul, Gradients<T>::pointMul_grad,
            recLen, indA, indB, rec.nodes[indA].value.shape, rec.nodes[indA].value.shapeLen);
    }
    else {
        size_t newLen = max(rec.nodes[indA].value.shapeLen, rec.nodes[indB].value.shapeLen);
        int* newShape = new int[newLen];
        combine_flexible(rec.nodes[indA].value.shape, rec.nodes[indA].value.shapeLen,
            rec.nodes[indB].value.shape, rec.nodes[indB].value.shapeLen,
            newShape, newLen);

        rec.nodes.emplace_back(Operations<T>::flexMul, Gradients<T>::flexMul_grad,
            recLen, indA, indB, newShape, newLen);
    }
    return recLen;
}

template <typename T>
int div(Recorder<T>& rec, int indA, int indB) {
    int recLen = rec.nodes.size();
    bool A_scalar = rec.nodes[indA].value.shapeLen == 0 && rec.nodes[indA].value.shape[0] == 1;
    bool B_scalar = rec.nodes[indB].value.shapeLen == 0 && rec.nodes[indB].value.shape[0] == 1;

    if (A_scalar || B_scalar) {
        int tensor = A_scalar ? indB : indA;
        int scalar = A_scalar ? indA : indB;
        
        rec.nodes.emplace_back(Operations<T>::scalarDiv, Gradients<T>::scalarDiv_grad,
            recLen, tensor, scalar, rec.nodes[tensor].value.shape, rec.nodes[tensor].value.shapeLen);
    }
    else if (rec.nodes[indA].value.shapeLen == rec.nodes[indB].value.shapeLen &&
            equal(rec.nodes[indA].value.shape, rec.nodes[indA].value.shape + rec.nodes[indA].value.shapeLen,
            rec.nodes[indB].value.shape)) {
        
        rec.nodes.emplace_back(Operations<T>::pointDiv, Gradients<T>::pointDiv_grad,
            recLen, indA, indB, rec.nodes[indA].value.shape, rec.nodes[indA].value.shapeLen);
    }
    else {
        size_t newLen = max(rec.nodes[indA].value.shapeLen, rec.nodes[indB].value.shapeLen);
        int* newShape = new int[newLen];
        combine_flexible(rec.nodes[indA].value.shape, rec.nodes[indA].value.shapeLen,
            rec.nodes[indB].value.shape, rec.nodes[indB].value.shapeLen,
            newShape, newLen);

        rec.nodes.emplace_back(Operations<T>::flexDiv, Gradients<T>::flexDiv_grad,
            recLen, indA, indB, newShape, newLen);
    }
    return recLen;
}

template <typename T>
int pow(Recorder<T>& rec, int indA, int indB) {
    int recLen = rec.nodes.size();
    bool A_scalar = rec.nodes[indA].value.shapeLen == 0 && rec.nodes[indA].value.shape[0] == 1;
    bool B_scalar = rec.nodes[indB].value.shapeLen == 0 && rec.nodes[indB].value.shape[0] == 1;

    if (A_scalar || B_scalar) {
        int tensor = A_scalar ? indB : indA;
        int scalar = A_scalar ? indA : indB;
        
        rec.nodes.emplace_back(Operations<T>::scalarPow, Gradients<T>::scalarPow_grad,
            recLen, tensor, scalar, rec.nodes[tensor].value.shape, rec.nodes[tensor].value.shapeLen);
    }
    else if (rec.nodes[indA].value.shapeLen == rec.nodes[indB].value.shapeLen &&
            equal(rec.nodes[indA].value.shape, rec.nodes[indA].value.shape + rec.nodes[indA].value.shapeLen,
            rec.nodes[indB].value.shape)) {
        
        rec.nodes.emplace_back(Operations<T>::pointPow, Gradients<T>::pointPow_grad,
            recLen, indA, indB, rec.nodes[indA].value.shape, rec.nodes[indA].value.shapeLen);
    }
    else {
        size_t newLen = max(rec.nodes[indA].value.shapeLen, rec.nodes[indB].value.shapeLen);
        int* newShape = new int[newLen];
        combine_flexible(rec.nodes[indA].value.shape, rec.nodes[indA].value.shapeLen,
            rec.nodes[indB].value.shape, rec.nodes[indB].value.shapeLen,
            newShape, newLen);

        rec.nodes.emplace_back(Operations<T>::flexPow, Gradients<T>::flexPow_grad,
            recLen, indA, indB, newShape, newLen);
    }
    return recLen;
}

template <typename T>
int negative(Recorder<T>& rec, int indA) {
    int recLen = rec.nodes.size();
    rec.nodes.emplace_back(Operations<T>::_negate, Gradients<T>::negate_grad, recLen, indA, -1, rec.nodes[indA].value.shape, rec.nodes[indA].value.shapeLen);
    return recLen;
}

template <typename T>
int square(Recorder<T>& rec, int indA) {
    int recLen = rec.nodes.size();
    rec.nodes.emplace_back(Operations<T>::sqr, Gradients<T>::sqr_grad, recLen, indA, -1, rec.nodes[indA].value.shape, rec.nodes[indA].value.shapeLen);
    return recLen;
}

template <typename T>
int sqrt(Recorder<T>& rec, int indA) {
    int recLen = rec.nodes.size();
    rec.nodes.emplace_back(Operations<T>::_sqrt, Gradients<T>::_sqrt_grad, recLen, indA, -1, rec.nodes[indA].value.shape, rec.nodes[indA].value.shapeLen);
    return recLen;
}

template <typename T>
int log(Recorder<T>& rec, int indA) {
    int recLen = rec.nodes.size();
    rec.nodes.emplace_back(Operations<T>::_log, Gradients<T>::_log_grad, recLen, indA, -1, rec.nodes[indA].value.shape, rec.nodes[indA].value.shapeLen);
    return recLen;
}

template <typename T>
int exp(Recorder<T>& rec, int indA) {
    int recLen = rec.nodes.size();
    rec.nodes.emplace_back(Operations<T>::_exp, Gradients<T>::_exp_grad, recLen, indA, -1, rec.nodes[indA].value.shape, rec.nodes[indA].value.shapeLen);
    return recLen;
}

template <typename T>
int abs(Recorder<T>& rec, int indA) {
    int recLen = rec.nodes.size();
    rec.nodes.emplace_back(Operations<T>::_abs, Gradients<T>::_abs_grad, recLen, indA, -1, rec.nodes[indA].value.shape, rec.nodes[indA].value.shapeLen);
    return recLen;
}

template <typename T>
int matmul(Recorder<T>& rec, int indA, int indB) {
    int recLen = rec.nodes.size();
    
    Tensor<T>& A = rec.nodes[indA].value;
    Tensor<T>& B = rec.nodes[indB].value;
    size_t newLen = max(rec.nodes[indA].value.shapeLen, rec.nodes[indB].value.shapeLen);
    int* newShape = new int[newLen];
    combine_matrix(rec.nodes[indA].value.shape, rec.nodes[indA].value.shapeLen,
        rec.nodes[indB].value.shape, rec.nodes[indB].value.shapeLen,
        newShape, newLen);

    if (newLen == 2) {
        rec.nodes.emplace_back(Operations<T>::matmul, Gradients<T>::matmul_grad, recLen, indA, indB, newShape, newLen);
    }
    else {
        rec.nodes.emplace_back(Operations<T>::batch_matmul, Gradients<T>::batch_matmul_grad, recLen, indA, indB, newShape, newLen);
    }
    
    return recLen;
}

template <typename T>
int sum(Recorder<T>& rec, int indA) {
    int recLen = rec.nodes.size();
    
    int newShape[] = {1};
    rec.nodes.emplace_back(Operations<T>::sum, Gradients<T>::sum_grad, recLen, indA, -1, newShape, 1);

    return recLen;
}

template <typename T>
int sum(Recorder<T>& rec, int indA, int dim) {
    int recLen = rec.nodes.size();

    size_t newLen = rec.nodes[indA].value.shapeLen - 1;
    int* newShape = new int[newLen];
    copy(rec.nodes[indA].value.shape, rec.nodes[indA].value.shape + dim, newShape);
    for (int i = dim; i < newLen; i++) {
        newShape[i] = rec.nodes[indA].value.shape[i + 1];
    }
    
    // save dim into node
    Tensor<T> temp({dim}, 0);
    rec.nodes.emplace_back(move(temp), recLen++);
    

    rec.nodes.emplace_back(Operations<T>::sum_dim, Gradients<T>::sum_dim_grad, recLen, indA, recLen - 1, newShape, newLen);

    return recLen;
}

template <typename T>
int mean(Recorder<T>& rec, int indA) {
    int recLen = rec.nodes.size();
    
    int newShape[] = {1};
    rec.nodes.emplace_back(Operations<T>::mean, Gradients<T>::mean_grad, recLen, indA, -1, newShape, 1);

    return recLen;
}

template <typename T>
int mean(Recorder<T>& rec, int indA, int dim) {
    int recLen = rec.nodes.size();

    size_t newLen = rec.nodes[indA].value.shapeLen - 1;
    int* newShape = new int[newLen];
    copy(rec.nodes[indA].value.shape, rec.nodes[indA].value.shape + dim, newShape);
    for (int i = dim; i < newLen; i++) {
        newShape[i] = rec.nodes[indA].value.shape[i + 1];
    }
    
    // save dim into node
    Tensor<T> temp({dim}, 0);
    rec.nodes.emplace_back(move(temp), recLen++);
    

    rec.nodes.emplace_back(Operations<T>::mean_dim, Gradients<T>::mean_dim_grad, recLen, indA, recLen - 1, newShape, newLen);

    return recLen;
}

template <typename T>
int transpose(Recorder<T>& rec, int indA, int dim) {
    int recLen = rec.nodes.size();

    int* newShape = new int[rec.nodes[indA].value.shapeLen];
    transp(newShape, rec.nodes[indA].value.shapeLen);

    rec.nodes.emplace_back(Operations<T>::transp, Gradients<T>::transp_dim, recLen, indA, -1, newShape, rec.nodes[indA].value.shapeLen);

    return recLen;
}