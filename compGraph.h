#pragma once

#include "tensor.h"
#include "gradients.h"

#include <array>
#include <utility>
#include <memory>

template <typename T>
struct INode {

    INode<T>* in1 = nullptr;

    Tensor<T> value;
    Tensor<T> gradient;

    bool evaluated = false;
    bool hasInputs = false;

    template <typename... Args>
    INode(int in1_idx, Args&&... args)
    : in1(in1_idx), evaluated(false), value(forward<Args>(args)...), hasInputs(true), gradient(value) {
        fill(value.val, value.val + value.len, 0);
        fill(gradient.val, gradient.val + gradient.len, 0.0);
    }

    virtual ~INode() = default;
    
    inline void reset() {
        fill(value.val, value.val + value.len, 0);
        fill(gradient.val, gradient.val + gradient.len, 0);
    }

    inline virtual void eval(const T* A, const T* B, T* C) = 0;
    inline virtual void grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC) = 0;
};

template <typename T>
struct Node_valued {
    
    Node_valued(Tensor<T>&& tensor) {
        this->evaluated = true;
        this-> hasInputs = false;
        this->value(move(tensor));
        this->gradient(this->value);
        fill(this->gradient.val, this->gradient.val + this->gradient.len, 0.0);
    }

};

template <typename T>
struct Node_unary {
    tensorOp<T> op = nullptr;
    gradientOp<T> grad_op = nullptr;

    size_t len[2] = { 0, 0 };

    template <typename... Args>
    Node_unary(tensorOp<T> operation, gradientOp<T> derivative, int in1_idx, Args&&... args)
    : op(operation), grad_op(derivative), INode<T>(in1_idx, args...) {}

    inline void eval(const T* A, const T* B, T* C) override {
        if (!this->evaluated) {
            this->in1.eval();
            this->in2.eval();

            op(this->in1.value.val, this->in2.value.val, this->value.val, nullptr, nullptr, nullptr, nullptr, nullptr, len[0]);
            this->evaluated = true;
        }
    }

    inline void grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC) override {
        grad_op(this->in1.value.val, this->in2.value.val, this->value.val, this->in1.gradient.val, this->in2.gradient.val, this->gradient.val,
            nullptr, nullptr, nullptr, nullptr, nullptr, len+1);

        if (this->in1.hasInputs) {
            this->in1.grad();
        }
        if (this->in2.hasInputs) {
            this->in2.grad();
        }
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
        grad_op(A, B, C, dA,dB, dC, strideA+1, strideB+1, strideC+1, reps+1, count+1, strideLen+1);
    }
};

template <typename T>
struct Node_binary : INode<T> {
    INode<T>* in2 = nullptr;

    tensorOp<T> op = nullptr;
    gradientOp<T> grad_op = nullptr;

    size_t len[2] = { 0, 0 };

    template <typename... Args>
    Node_binary(tensorOp<T> operation, gradientOp<T> derivative, int in1_idx, int in2_idx, Args&&... args)
    : in2(in2_idx), op(operation), grad_op(derivative), INode<T>(in1_idx, args...) {}

    inline void eval(const T* A, const T* B, T* C) override {
        op(A, B, C, nullptr, nullptr, nullptr, nullptr, nullptr, len[0]);
        this->evaluated = true;
    }

    inline void grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC) override {
        grad_op(A, B, C, dA, dB, dC, nullptr, nullptr, nullptr, nullptr, nullptr, len[1]);
    }
};

template <typename T>
struct Node_binary_flex : INode<T> {
    INode<T>* in2 = nullptr;

    tensorOp<T> op = nullptr;
    gradientOp<T> grad_op = nullptr;

    size_t nEntries = 0;
    int** strideA = nullptr;
    int** strideB = nullptr;
    int** strideC = nullptr;
    int** reps = nullptr;
    int** count = nullptr;
    size_t* strideLen = nullptr;

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
   vector<unique_ptr<INode<T>>> nodes;

   INode<T>* append(Tensor<T>&& tensor) {
       auto ptr = std::make_unique<Node_valued<T>>(move(tensor));
       INode<T>* raw = ptr.get();
       nodes.emplace_back(move(ptr));
       return raw;
   } 

    // evaluate tensors in CompGraph at index specified
    // returns reference array to Tensors in given order
    template<typename... ptrs>
    array<Tensor<T>*, sizeof...(ptrs)> evaluate(ptrs... nodes) {
        int node_arr[] = {nodes...};
        array<Tensor<T>*, sizeof...(nodes)> values;
        for (int i = 0; i < sizeof...(nodes); i++) {
            node_arr[i]->eval();
            values[i] = &(node_arr[i]->value);
        }

        return values;
    }

    // calculates partial derivatives for all Node going back from f
    // returns reference to derivative of f w.r.t. x
    template<typename... ptrs>
    array<Tensor<T>*, sizeof...(ptrs)> getGradient(INode<T>* df_idx, ptrs... dx_idx) {
        INode<T>& f = *df_idx;
        fill(f.gradient.val, f.gradient.val + f.gradient.len, 1.0);
    
        //grad(df_idx);
        f.grad();
    
        INode<T>* node_arr[] = {dx_idx...};
        array<Tensor<T>*, sizeof...(dx_idx)> partials;
        for (size_t i = 0; i < sizeof...(dx_idx); i++) {
            partials[i] = &(node_arr[i]->gradient);
        }

        return partials;
    }

    void reset() {
        for (INode<T>* node : nodes) {
            node->reset();
        }
    }

    private:
    tView<T> eval(int index) {
        INode<T>& node = *nodes[index];
        if (node.evaluated) return node.value.view();
    
        tView<T> val1 = this->eval(node.in1);
        tView<T> val2 = node.in2 < 0 ? tView<T>() : this->eval(node.in2);
        tView<T> out = node.value.view();
    
        node.eval(val1.val, val2.val, node.value.val);
    
        return out;
    }

    void grad(int index) {
        INode<T>& node = *nodes[index];

        tView<T> value = node.value.view();
        tView<T> seed = node.gradient.view();
    
        tView<T> val1 = nodes[node.in1].value.view();
        tView<T> grad1 = nodes[node.in1].gradient.view();
        int in2 = node.in2;
        tView<T> val2 = in2 < 0 ? tView<T>() : nodes[in2].value.view();
        tView<T> grad2 = in2 < 0 ? tView<T>() : nodes[in2].gradient.view();
    
        node.grad(nodes[node.in1].value.val, nodes[node.in2].value.val, node.value.val, nodes[node.in1].gradient.val, nodes[node.in2].gradient.val, node.gradient.val);
    
        if (nodes[node.in1].hasInputs) {
            grad(node.in1);
        }
        if (nodes[in2].hasInputs && in2 >= 0) {
            grad(in2);
        }
    }
};
