#pragma once

#include "../tensor/tensor.hpp" // for Tensor
#include <array>             // for std::array
#include <cstddef>           // for size_t
#include <memory>            // for std::unique_ptr, std::make_unique
#include <utility>           // for std::forward
#include <vector>            // for std::vector

namespace kaad {
template <typename T> struct INode;
template <typename T> struct Node_valued;

/**
 * @brief Represents a computation graph for automatic differentiation.
 *
 * This graph stores nodes derived from the INode<T> interface. It supports
 * evaluating node values and computing gradients through backpropagation.
 *
 * @tparam T The data type of the tensor values.
 */
template <typename T> struct CompGraph {
    std::vector<std::unique_ptr<INode<T>>>
        nodes; ///< Holds unique pointers pointing to computation nodes

    /**
     * @brief Constructs a Node valued with the given tensor arguments and
     * appends it to the node list.
     *
     * Creates a Tensor<T> using the forwarded constructor arguments, wraps it
     * in a Node_valued<T>, and stores the node as a std::unique_ptr in the
     * `nodes` container.
     *
     * @tparam Args Variadic template parameter pack for tensor constructor
     * arguments.
     *
     * @param tensor_args Arguments forwarded to the Tensor<T> constructor.
     * @return A raw pointer to the newly created Node_valued<T>.
     */
    template <typename... Args> INode<T> *append(Args &&...tensor_args) {
        Tensor<T> tensor(std::forward<Args>(tensor_args)...);
        auto ptr = std::make_unique<Node_valued<T>>(std::move(tensor));
        INode<T> *raw = ptr.get();
        nodes.emplace_back(std::move(ptr));
        return raw;
    }

    /**
     * @brief Evaluates a list of nodes and returns their tensor values.
     *
     * Accepts a variadic list of node pointers, evaluates each one by calling
     * its `eval()` method, and returns a std::array of pointers to their
     * resulting Tensor<T> values in the same order.
     *
     * @tparam ptrs Variadic template parameter pack of INode<T>* types.
     *
     * @param node_ptrs A list of pointers to nodes to be evaluated.
     * @return An array of Tensor<T>* pointers, each corresponding to the value
     * of the evaluated node.
     */
    template <typename... ptrs>
    std::array<Tensor<T> *, sizeof...(ptrs)> evaluate(ptrs... node_ptrs) {
        INode<T> *node_arr[] = {node_ptrs...};
        std::array<Tensor<T> *, sizeof...(node_ptrs)> values;
        for (int i = 0; i < sizeof...(node_ptrs); i++) {
            node_arr[i]->eval();
            values[i] = &(node_arr[i]->value);
        }

        return values;
    }

    /**
     * @brief Computes gradients of the computation graph with respect to the
     * given input nodes.
     *
     * Initializes the gradient of the output node `df_ptr` to 1 and performs
     * backpropagation through the graph. Returns a list of pointers to the
     * gradient tensors corresponding to each input node in `dx_ptrs`.
     *
     * @tparam ptrs Variadic template parameter pack of INode<T>* types.
     *
     * @param df_ptr Pointer to the output node (target function) with respect
     * to which gradients are computed.
     * @param dx_ptrs A list of input node pointers for which the gradients are
     * requested.
     * @return An array of Tensor<T>* pointers representing the gradients ∂f/∂xᵢ
     * for each input node.
     */
    template <typename... ptrs>
    std::array<Tensor<T> *, sizeof...(ptrs)> getGradient(INode<T> *df_ptr,
                                                         ptrs... dx_ptrs) {
        INode<T> &f = *df_ptr;
        std::fill(f.gradient.val.begin(), f.gradient.val.end(), 1.0);

        f.getGrad();

        INode<T> *node_arr[] = {dx_ptrs...};
        std::array<Tensor<T> *, sizeof...(dx_ptrs)> partials;
        for (size_t i = 0; i < sizeof...(dx_ptrs); i++) {
            partials[i] = &(node_arr[i]->gradient);
        }

        return partials;
    }

    /**
     * @brief Resets all node values and gradients in the computation graph to
     * zero.
     *
     * Calls the `reset()` method on each node in the graph, zeroing out their
     * associated value and gradient tensors. This is typically used before a
     * new forward pass.
     */
    void reset() {
        for (int i = 0; i < nodes.size(); i++) {
            nodes[i]->reset();
        }
    }
};
} // namespace kaad
