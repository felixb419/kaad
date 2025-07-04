#include <stddef.h>   // for size_t
#include <algorithm>  // for fill
#include <memory>     // for unique_ptr, make_unique
#include <vector>     // for vector
#include "tensor.h"   // for Tensor
template <typename T> struct INode;
template <typename T> struct Node_valued;

#pragma once

template <typename T>
struct CompGraph {
   vector<unique_ptr<INode<T>>> nodes;

    INode<T>* append(Tensor<T>&& tensor) {
        auto ptr = std::make_unique<Node_valued<T>>(move(tensor));
        INode<T>* raw = ptr.get();
        nodes.emplace_back(move(ptr));
        return raw;
    } 

    // evaluate nodes specified by node_ptrs and output array of values in order
    template<typename... ptrs>
    array<Tensor<T>*, sizeof...(ptrs)> evaluate(ptrs... node_ptrs) {
        INode<T>* node_arr[] = {node_ptrs...};
        array<Tensor<T>*, sizeof...(node_ptrs)> values;
        for (int i = 0; i < sizeof...(node_ptrs); i++) {
            node_arr[i]->eval();
            values[i] = &(node_arr[i]->value);
        }

        return values;
    }

    // compute gradients of Nodes in graph w.r.t. df and output array of gradients of dx
    template<typename... ptrs>
    array<Tensor<T>*, sizeof...(ptrs)> getGradient(INode<T>* df_ptr, ptrs... dx_ptrs) {
        INode<T>& f = *df_ptr;
        fill(f.gradient.val, f.gradient.val + f.gradient.len, 1.0);
    
        f.getGrad();
    
        INode<T>* node_arr[] = {dx_ptrs...};
        array<Tensor<T>*, sizeof...(dx_ptrs)> partials;
        for (size_t i = 0; i < sizeof...(dx_ptrs); i++) {
            partials[i] = &(node_arr[i]->gradient);
        }

        return partials;
    }

    void reset() {
        for (INode<T>* node : nodes) {
            node->reset();
        }
    }
};
