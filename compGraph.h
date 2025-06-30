#pragma once

#include "nodes.h"

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
        INode<T>* node_arr[] = {nodes...};
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
    
        f.getGrad();
    
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
