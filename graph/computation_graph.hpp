#pragma once

#include "../tensor/tensor.hpp" // for Tensor
#include "node_handle.hpp"      // for Node_handle
#include "nodes/inode.hpp"      // for INode
#include "nodes/input.hpp"      // for InputNode
#include <array>                // for std::array
#include <cstddef>              // for std::size_t
#include <memory>               // for std::unique_ptr, std::make_unique
#include <vector>               // for std::vector

namespace kaad {

/**
 * @brief Represents a computation graph for automatic differentiation.
 *
 * This graph stores nodes derived from the INode interface. It supports
 * evaluating node values and computing gradients through backpropagation.
 */
class Computation_graph {
  public:
    std::vector<std::unique_ptr<INode>>
        nodes; ///< Holds unique pointers pointing to computation nodes

    /**
     * @brief Returns a pointer to a node handle.
     * @warning Will throw argument_error if @p node doesnt refer to
     * correct graph.
     * @param node Node handle of the relevant node.
     * @return Pointer to the Node.
     */
    INode *get_node(Node_handle node);

    /**
     * @brief Get the last node in the graph.
     * @return Handle of the node at the back of the node vector.
     */
    Node_handle back_handle() noexcept;

    /**
     * @brief Constructs an evaluated node with no inputs and adds it to the
     * graph.
     *
     * Creates a Tensor using the given shape, wraps it in a InputNode, and
     * stores the node as a std::unique_ptr in the `nodes` container.
     *
     * @return A handle of the newly created InputNode.
     */
    Node_handle add_input_node(std::span<const int> value_shape);

    /**
     * @brief Evaluates a list of nodes and returns their tensor values.
     *
     * Accepts a variadic list of node handles, evaluates each one by calling
     * its `eval()` method, and returns a std::array of pointers to their
     * resulting Tensor values in the same order.
     *
     * @tparam Node_handles Variadic template parameter pack of Node_handle
     * types.
     *
     * @param nodes A list of handles of nodes to be evaluated.
     * @return An array of Tensor* pointers, each corresponding to the value
     * of the evaluated node.
     */
    template <typename... Node_handles>
        requires(std::same_as<Node_handles, Node_handle> && ...)
    std::array<Tensor *, sizeof...(Node_handles)>
    evaluate(Node_handles... nodes) {

        Node_handle node_arr[] = {nodes...};
        std::array<Tensor *, sizeof...(nodes)> values;

        for (int i = 0; i < sizeof...(nodes); i++) {
            INode *node_ptr = this->get_node(node_arr[i]);
            node_ptr->eval();
            values[i] = &node_ptr->value();
        }

        return values;
    }

    /**
     * @brief Computes gradients of the computation graph with respect to the
     * given input nodes.
     *
     * Initializes the gradient of the output node `output` to 1 and performs
     * backpropagation through the graph. Returns a list of pointers to the
     * gradient tensors corresponding to each input node in `wrt`.
     *
     * @tparam ptrs Variadic template parameter pack of Node_handle types.
     *
     * @param output Handle of the output node (target function) with respect
     * to which gradients are computed.
     * @param wrt A list of input node handles for which the gradients are
     * requested.
     * @return An array of Tensor* pointers representing the gradients ∂f/∂xᵢ
     * for each input node.
     */
    template <typename... Handles>
        requires(std::same_as<Handles, Node_handle> && ...)
    std::array<Tensor *, sizeof...(Handles)> getGradient(Node_handle output,
                                                         Handles... wrt) {
        INode *f = this->get_node(output);
        std::fill(f->gradient().elements_.begin(),
                  f->gradient().elements_.end(), 1.0);

        f->getGrad();

        std::array<Node_handle, sizeof...(wrt)> nodes = {wrt...};
        std::array<Tensor *, sizeof...(wrt)> partials = {};
        for (std::size_t i = 0; i < sizeof...(wrt); i++) {
            partials[i] = &this->get_node(nodes[i])->gradient();
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
    void reset();
};

std::ostream &operator<<(std::ostream &os, Node_handle node);

} // namespace kaad
