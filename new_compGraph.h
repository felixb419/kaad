#pragma once

#include "tensor.h"
#include "gradients.h"

#include <array>
#include <utility>

template <typename T>
struct INode {

    int in1;

    Tensor<T> value;
    Tensor<T> gradient;

    bool evaluated = false;
    bool hasInputs = false;

    // construct as evaluated
    INode(Tensor<T> && tensor)
    : in1(-1), evaluated(true), value(move(tensor)), hasInputs(false), gradient(value) {
        fill(gradient.val, gradient.val + gradient.len, 0.0);
    }

    // construct as to be evaluated
    template <typename... Args>
    INode(int in1_idx, Args&&... args)
    : in1(in1_idx), evaluated(false), value(forward<Args>(args)...), hasInputs(true), gradient(value) {}
    
    inline void reset() {
        fill(value.val, value.val + value.len, 0);
        fill(gradient.val, gradient.val + gradient.len, 0);
    }

    inline virtual void eval(const T* A, const T* B, T* C) = 0;
    inline virtual void grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC) = 0;
};

template <typename T>
struct Node_unary {
    tensorOp<T> op = nullptr;
    gradientOp<T> grad_op = nullptr;

    size_t* len = nullptr;

    // construct as evaluated
    Node_unary(Tensor<T> && tensor)
    : op(nullptr), grad_op(nullptr), INode<T>(move(tensor)) {}

    // construct as to be evaluated
    template <typename... Args>
    Node_unary(tensorOp<T> operation, gradientOp<T> derivative, int in1_idx, Args&&... args)
    : op(operation), grad_op(derivative), INode<T>(in1_idx, args...) {}

    ~Node_unary() {
        delete[] len;
    }

    inline void eval(const T* A, const T* B, T* C) override {
        op(A, B, C, nullptr, nullptr, nullptr, nullptr, nullptr, len[0]);
        this->evaluated = true;
    }

    inline void grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC) override {
        grad_op(A, B, C, dA, dB, dC, nullptr, nullptr, nullptr, nullptr, nullptr, len+1);
    }
};

template <typename T>
struct Node_unary_flex {
    tensorOp<T> op = nullptr;
    gradientOp<T> grad_op = nullptr;

    size_t nEntries = 0;
    int** strideA = nullptr;
    int** strideB = nullptr;
    int** strideC = nullptr;
    int** reps = nullptr;
    int** count = nullptr;
    size_t* strideLen = nullptr;

    // construct as evaluated
    Node_unary_flex(Tensor<T> && tensor)
    : op(nullptr), grad_op(nullptr), INode<T>(move(tensor)) {}

    // construct as to be evaluated
    template <typename... Args>
    Node_unary_flex(tensorOp<T> operation, gradientOp<T> derivative, int in1_idx, Args&&... args)
    : op(operation), grad_op(derivative), INode<T>(in1_idx, args...) {}

    ~Node_unary_flex() {
        for (int i = 0; i < nEntries; i++) {
            delete[] strideA[i];
            delete[] strideB[i];
            delete[] strideC[i];
            delete[] reps[i];
            delete[] count[i];
        }
        delete[] strideLen;
    }

    inline void eval(const T* A, const T* B, T* C) override {
        op(A, B, C, strideA[0], strideB[0], strideC[0], reps[0], count[0], strideLen[0]);
        this->evaluated = true;
    }

    inline void grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC) override {
        grad_op(A, B, C, dA, dB, dC, strideA+1, strideB+1, strideC+1, reps+1, count+1, strideLen+1);
    }
};

template <typename T>
struct Node_binary : INode<T> {
    int in2 = -1;

    tensorOp<T> op = nullptr;
    gradientOp<T> grad_op = nullptr;

    size_t* len = nullptr;

    // construct as evaluated
    Node_binary(Tensor<T> && tensor)
    : in2(-1), op(nullptr), grad_op(nullptr), INode<T>(move(tensor)) {}

    // construct as to be evaluated
    template <typename... Args>
    Node_binary(tensorOp<T> operation, gradientOp<T> derivative, int in1_idx, int in2_idx, Args&&... args)
    : in2(in2_idx), op(operation), grad_op(derivative), INode<T>(in1_idx, args...) {}

    ~Node_binary() {
        delete[] len;
    }

    inline void eval(const T* A, const T* B, T* C) override {
        op(A, B, C, nullptr, nullptr, nullptr, nullptr, nullptr, len[0]);
        this->evaluated = true;
    }

    inline void grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC) override {
        grad_op(A, B, C, dA, dB, dC, nullptr, nullptr, nullptr, nullptr, nullptr, len+1);
    }
};

template <typename T>
struct Node_binary_flex : INode<T> {
    int in2 = -1;

    tensorOp<T> op = nullptr;
    gradientOp<T> grad_op = nullptr;

    size_t nEntries = 0;
    int** strideA = nullptr;
    int** strideB = nullptr;
    int** strideC = nullptr;
    int** reps = nullptr;
    int** count = nullptr;
    size_t* strideLen = nullptr;

    // construct as evaluated
    Node_binary_flex(Tensor<T> && tensor)
    : in2(-1), op(nullptr), grad_op(nullptr), INode<T>(move(tensor)) {}

    // construct as to be evaluated
    template <typename... Args>
    Node_binary_flex(tensorOp<T> operation, gradientOp<T> derivative, int in1_idx, int in2_idx, Args&&... args)
    : in2(in2_idx), op(operation), grad_op(derivative), INode<T>(in1_idx, args...) {}

    ~Node_binary_flex() {
        for (int i = 0; i < nEntries; i++) {
            delete[] strideA[i];
            delete[] strideB[i];
            delete[] strideC[i];
            delete[] reps[i];
            delete[] count[i];
        }
        delete[] strideLen;
    }

    inline void eval(const T* A, const T* B, T* C) override {
        op(A, B, C, strideA[0], strideB[0], strideC[0], reps[0], count[0], strideLen[0]);
        this->evaluated = true;
    }

    inline void grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC) override {
        grad_op(A, B, C, dA, dB, dC, strideA+1, strideB+1, strideC+1, reps+1, count+1, strideLen+1);
    }
};

template <typename T>
struct CompGraph {
   vector<INode<T>*> nodes;

   int addNode(Tensor<T>&& tensor) {
   }
};
