#pragma once

#include "tensor.h"

#include <array>
#include <utility>

template <typename T>
using tensorOP = void(*)(const tView<T>* in1, const tView<T>* in2, tView<T>* out);
template <typename T>
using gradientOP = void(*)(const tView<T>* seed, tView<T>* in1, tView<T>* d_in1, tView<T>* in2, tView<T>* d_in2, tView<T>* res);

template <typename T>
struct Node {
    public:
        int in1;
        int in2;

        tensorOP<T> op;
        gradientOP<T> grad_op;

        bool evaluated;
        Tensor<T> value;

        bool hasInputs;
        Tensor<T> gradient;

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
        template<size_t N>
        array<Tensor<T>*, N> getGradient(int f_ind, int (&x_ind)[N]) {

            fill(nodes[f_ind].gradient.val, nodes[f_ind].gradient.val + nodes[f_ind].gradient.len, 1.0);
        
            grad(f_ind);
        
            array<Tensor<T>*, N> partials;
            for (size_t i = 0; i < N; i++) {
                partials[i] = &nodes[x_ind[i]].gradient;
            }

            return partials;
        }

    private:
        tView<T> eval(int index) {
            Node<T>& cur = nodes[index];
            if (cur.evaluated) return cur.value.view();
        
            tView<T> val1 = this->eval(cur.in1);
            tView<T> val2 = cur.in2 < 0 ? tView<T>() : this->eval(cur.in2);
            tView<T> out = cur.value.view();
        
            cur.op(&val1, &val2, &out);
            cur.evaluated = true;
        
            return out;
        }

        void grad(int index) {
            tView<T> value = nodes[index].value.view();
            tView<T> seed = nodes[index].gradient.view();
        
            tView<T> val1 = nodes[nodes[index].in1].value.view();
            tView<T> grad1 = nodes[nodes[index].in1].gradient.view();
            int in2 = nodes[index].in2;
            tView<T> val2 = in2 < 0 ? tView<T>() : nodes[in2].value.view();
            tView<T> grad2 = in2 < 0 ? tView<T>() : nodes[in2].gradient.view();
        
            nodes[index].grad_op(&seed, &val1, &grad1, &val2, &grad2, &value);
        
            if (nodes[nodes[index].in1].hasInputs) {
                grad(nodes[index].in1);
            }
            if (nodes[in2].hasInputs && in2 >= 0) {
                grad(in2);
            }
        }
};
