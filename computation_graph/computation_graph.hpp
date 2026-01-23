#pragma once

#include "../tensor/tensor.hpp" // for Tensor
#include "node_handle.hpp"      // for Node_handle
#include <array>                // for std::array
#include <cstddef>              // for size_t
#include <memory>               // for std::unique_ptr, std::make_unique
#include <utility>              // for std::forward
#include <vector>               // for std::vector

namespace kaad {
template <typename T> class INode;
template <typename T> class Node_valued;

/**
 * @brief Represents a computation graph for automatic differentiation.
 *
 * This graph stores nodes derived from the INode<T> interface. It supports
 * evaluating node values and computing gradients through backpropagation.
 *
 * @tparam T The data type of the tensor values.
 */
template <typename T> class Computation_graph {
  public:
    std::vector<std::unique_ptr<INode<T>>>
        nodes; ///< Holds unique pointers pointing to computation nodes

    /**
     * @brief Returns a pointer to a node handle.
     * @warning Will throw std::invalid_argument if @p node doesnt refer to
     * correct graph.
     * @param node Node handle of the relevant node.
     * @return Pointer to the Node.
     */
    INode<T> *get_node(Node_handle<T> node) {
        if (node.origin_ != this) {
            throw std::invalid_argument(
                "node does not belong to this instance of Computation_graph");
        }
        if (node.idx_ < 0 && node.idx_ >= this->nodes.size()) {
            throw std::invalid_argument(
                std::to_string(node.idx_) +
                "is not a valid index for this Computation_graph");
        }

        return this->nodes[node.idx_].get();
    }

    /**
     * @brief Get the last node in the graph.
     * @return Handle of the node at the back of the node vector.
     */
    Node_handle<T> back_handle() {
        return Node_handle<T>(this->nodes.size() - 1, this);
    }

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
     * @return A handle of the newly created Node_valued<T>.
     */
    template <typename... Args> Node_handle<T> append(Args &&...tensor_args) {
        this->nodes.push_back(std::make_unique<Node_valued<T>>(
            std::forward<Args>(tensor_args)...));

        return Node_handle<T>(this->nodes.size() - 1, this);
    }

    /**
     * @brief Evaluates a list of nodes and returns their tensor values.
     *
     * Accepts a variadic list of node handles, evaluates each one by calling
     * its `eval()` method, and returns a std::array of pointers to their
     * resulting Tensor<T> values in the same order.
     *
     * @tparam Node_handles Variadic template parameter pack of Node_handle
     * types.
     *
     * @param nodes A list of handles of nodes to be evaluated.
     * @return An array of Tensor<T>* pointers, each corresponding to the value
     * of the evaluated node.
     */
    template <typename... Node_handles>
        requires(std::same_as<Node_handles, Node_handle<T>> && ...)
    std::array<Tensor<T> *, sizeof...(Node_handles)>
    evaluate(Node_handles... nodes) {

        Node_handle<T> node_arr[] = {nodes...};
        std::array<Tensor<T> *, sizeof...(nodes)> values;

        for (int i = 0; i < sizeof...(nodes); i++) {
            INode<T> *node_ptr = this->get_node(node_arr[i]);
            node_ptr->eval();
            values[i] = &node_ptr->value;
        }

        return values;
    }

    /**
     * @brief Computes gradients of the computation graph with respect to the
     * given input nodes.
     *
     * Initializes the gradient of the output node `df` to 1 and performs
     * backpropagation through the graph. Returns a list of pointers to the
     * gradient tensors corresponding to each input node in `dx`.
     *
     * @tparam ptrs Variadic template parameter pack of Node_handle<T> types.
     *
     * @param df Handle of the output node (target function) with respect
     * to which gradients are computed.
     * @param dx A list of input node handles for which the gradients are
     * requested.
     * @return An array of Tensor<T>* pointers representing the gradients ∂f/∂xᵢ
     * for each input node.
     */
    template <typename... Handles>
        requires(std::same_as<Handles, Node_handle<T>> && ...)
    std::array<Tensor<T> *, sizeof...(Handles)> getGradient(Node_handle<T> df,
                                                            Handles... dx) {
        INode<T> *f = this->get_node(df);
        std::fill(f->gradient.elements_.begin(), f->gradient.elements_.end(),
                  1.0);

        f->getGrad();

        std::array<Node_handle<T>, sizeof...(dx)> nodes = {dx...};
        std::array<Tensor<T> *, sizeof...(dx)> partials = {};
        for (size_t i = 0; i < sizeof...(dx); i++) {
            partials[i] = &this->get_node(nodes[i])->gradient;
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
