#pragma once

#include "tensor.h"

#include <array>
#include <utility>

template <typename T>
using tensorOP = void(*)(const T* A, const T* B, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen);
template <typename T>
using gradientOP = void(*)(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t stridelen);

template <typename T>
struct Node {
    public:
        int in1 = -1;
        int in2 = -1;

        tensorOP<T> op = nullptr;
        gradientOP<T> grad_op = nullptr;
        void* ctx = nullptr;

        bool evaluated = false;
        Tensor<T> value;

        bool hasInputs = false;
        Tensor<T> gradient;

        int* strideA = nullptr;
        int* strideB = nullptr;
        int* strideC = nullptr;
        int* reps = nullptr;
        int* count = nullptr;
        size_t strideLen;

        // construct as evaluated
        Node(Tensor<T> && tensor)
        : in1(-1), in2(-1), op(nullptr), grad_op(nullptr),
        evaluated(true), value(move(tensor)), hasInputs(false), gradient(value) {
            fill(gradient.val, gradient.val + gradient.len, 0.0);
        }

        // construct as to be evaluated
        template <typename... Args>
        Node(tensorOP<T> operation, gradientOP<T> derivative, int in1_index, int in2_index, Args&&... args)
        : in1(in1_index), in2(in2_index), op(operation), grad_op(derivative),
        evaluated(false), value(forward<Args>(args)...), hasInputs(true), gradient(value) {}

        ~Node() {
            delete[] strideA;
            delete[] strideB;
            delete[] strideC;
            delete[] static_cast<int*>(ctx);
        }

        inline void eval(const T* A, const T* B, T* C) {
            op(A, B, C, strideA, strideB, strideC, reps, count, strideLen);
            evaluated = true;
        }

        inline void grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC) {
            grad_op(A, B, C, dA, dB, dC, strideA, strideB, strideC, reps, count, strideLen);
        }
};

template <typename T>
ostream& operator<<(ostream& out, const Node<T>& self) {
    cout << "node at: " << self.index << endl;
    if (self.evaluated) {
        cout << "value:" << endl << self.value << endl;
        cout << "gradient:" << endl << self.gradient;
    }
    else {
        cout << "node not evaluated";
    }
    return out;
}

template <typename T>
class Recorder {
    public:
        vector<Node<T>> nodes;
    
        Recorder() {}

        int append(Tensor<T>&& tensor) {
            int idx = nodes.size();
            nodes.emplace_back(move(tensor));
            return idx;
        }

        // evaluate tensors in Recorder at index specified
        // returns reference array to Tensors in given order
        template<typename... ints>
        array<Tensor<T>*, sizeof...(ints)> evaluate(ints... indeces) {
            static_assert((is_same_v<ints, int> && ...), "all indeces args must be of type int");

            int inds[] = {indeces...};
            array<Tensor<T>*, sizeof...(indeces)> values;
            for (size_t i = 0; i < sizeof...(indeces); i++) {
                this->eval(inds[i]);
                values[i] = &nodes[inds[i]].value;
            }

            return values;
        }

        // calculates partial derivatives going back from f
        // returns reference to derivative of f w.r.t. x
        template<typename... ints>
        array<Tensor<T>*, sizeof...(ints)> getGradient(int df_idx, ints... dx_idx) {
            static_assert((is_same_v<ints, int> && ...), "all indeces args must be of type int");

            Node<T>& f = nodes[df_idx];
            fill(f.gradient.val, f.gradient.val + f.gradient.len, 1.0);
        
            grad(df_idx);
        
            int inds[] = {dx_idx...};
            array<Tensor<T>*, sizeof...(dx_idx)> partials;
            for (size_t i = 0; i < sizeof...(dx_idx); i++) {
                partials[i] = &nodes[inds[i]].gradient;
            }

            return partials;
        }

    private:
        tView<T> eval(int index) {
            Node<T>& node = nodes[index];
            if (node.evaluated) return node.value.view();
        
            tView<T> val1 = this->eval(node.in1);
            tView<T> val2 = node.in2 < 0 ? tView<T>() : this->eval(node.in2);
            tView<T> out = node.value.view();
        
            node.eval(val1.val, val2.val, node.value.val);
        
            return out;
        }

        void grad(int index) {
            Node<T>& node = nodes[index];

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
