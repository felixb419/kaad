#pragma once

#include <stddef.h>   // for size_t
#include <memory>     // for unique_ptr, make_unique
#include <vector>     // for vector
#include <array>
#include "tensor.h"   // for Tensor

namespace kaad {
    template <typename T> struct INode;
    template <typename T> struct Node_valued;

    template <typename T>
    struct CompGraph {
        std::vector<std::unique_ptr<INode<T>>> nodes;

        INode<T>* append(Tensor<T>&& tensor) {
            auto ptr = std::make_unique<Node_valued<T>>(std::move(tensor));
            INode<T>* raw = ptr.get();
            nodes.emplace_back(std::move(ptr));
            return raw;
        } 

        // evaluate nodes specified by node_ptrs and output array of values in order
        template<typename... ptrs>
        std::array<Tensor<T>*, sizeof...(ptrs)> evaluate(ptrs... node_ptrs) {
            INode<T>* node_arr[] = {node_ptrs...};
            std::array<Tensor<T>*, sizeof...(node_ptrs)> values;
            for (int i = 0; i < sizeof...(node_ptrs); i++) {
                node_arr[i]->eval();
                values[i] = &(node_arr[i]->value);
            }

            return values;
        }

        // compute gradients of Nodes in graph w.r.t. df and output array of gradients of dx
        template<typename... ptrs>
        std::array<Tensor<T>*, sizeof...(ptrs)> getGradient(INode<T>* df_ptr, ptrs... dx_ptrs) {
            INode<T>& f = *df_ptr;
            std::fill(f.gradient.val, f.gradient.val + f.gradient.len, 1.0);
        
            f.getGrad();
        
            INode<T>* node_arr[] = {dx_ptrs...};
            std::array<Tensor<T>*, sizeof...(dx_ptrs)> partials;
            for (size_t i = 0; i < sizeof...(dx_ptrs); i++) {
                partials[i] = &(node_arr[i]->gradient);
            }

            return partials;
        }

        void reset() {
            for (int i = 0; i < nodes.size(); i++) {
                nodes[i]->reset();
            }
        }
    };
}    
