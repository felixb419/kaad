#pragma once

#include "../scalar.hpp"        // for Scalar
#include "../tensor/tensor.hpp" // for Tensor
#include "nodes/inode.hpp"      // for INode
#include <algorithm>            // for fill
#include <array>                // for array
#include <cstddef>              // for size_t
#include <initializer_list>     // for initializer_list
#include <iosfwd>               // for ostream
#include <memory>               // for unique_ptr
#include <span>                 // for span
#include <vector>               // for vector

namespace kaad {

class Node_handle;

/**
 * @brief Represents a computation graph for automatic differentiation.
 *
 * This graph stores nodes derived from the INode interface. It supports
 * evaluating node values and computing gradients through backpropagation.
 */
class Computation_graph {
  private:
    std::vector<std::unique_ptr<INode>>
        nodes; ///< Holds unique pointers pointing to computation nodes

    /**
     * @brief Get the last node in the graph.
     * @return Handle of the node at the back of the node vector.
     */
    Node_handle back_handle() noexcept;

    /**
     * @brief Returns a pointer to a node handle.
     * @warning Will throw argument_error if @p node doesnt refer to
     * correct graph.
     * @param node Node handle of the relevant node.
     * @return Pointer to the Node.
     */
    INode *get_node(Node_handle node);

  public:
    /**
     * @brief Returns a const pointer to a node handle.
     * @warning Will throw argument_error if @p node doesnt refer to
     * correct graph.
     * @param node Node handle of the relevant node.
     * @return Immutable pointer to the Node.
     */
    const INode *get_node(Node_handle node) const;

    /**
     * @brief Constructs an evaluated node with no inputs and adds it to the
     * graph.
     *
     * Creates a Tensor using the given shape, wraps it in a InputNode, and
     * stores the node as a std::unique_ptr in the `nodes` container. Also sets
     * @p node_value_elements to the value array of the node so it can be filled
     * with values.
     *
     * @param[in] value_shape Shape of the tensor of the input node.
     * @param[out] node_value_elements Range of the value elements fo the node.
     * @return A handle of the newly created InputNode.
     */
    Node_handle add_input_node(std::span<const int> value_shape,
                               std::span<Scalar> &node_value_elements);

    /**
     * @brief Evaluates a list of nodes and returns their tensor values.
     * @warning reset has to be called before nodes are evaluated.
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
    std::vector<const Tensor *> evaluate(std::span<const Node_handle> nodes);

    /**
     * @brief Computes gradients of the computation graph with respect to the
     * given input nodes.
     * @warning evaluate has to be called on @p output before this.
     * @warning getGradient should only be called once before resetting and
     * evaluating again.
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
     * @return An array of Tensor* pointers representing the gradients df/dx
     * for each input node.
     */
    std::vector<const Tensor *>
    getGradient(Node_handle output, std::span<const Node_handle> inputs);

    /**
     * @brief Resets all node values and gradients in the computation graph.
     *
     * Calls the `reset()` method on each node in the graph, zeroing out their
     * associated value and gradient tensors. This is typically used before a
     * new forward pass.
     */
    void reset();

    friend std::ostream &operator<<(std::ostream &os, Node_handle node);

    friend class Node_handle;

    template <class Kernel>
    friend Node_handle binOperator(Computation_graph &rec, Node_handle A,
                                   Node_handle B, const char *opName);
    friend Node_handle dot(Computation_graph &rec, Node_handle A,
                           Node_handle B);
    friend Node_handle matmul(Computation_graph &rec, Node_handle A,
                              Node_handle B);
    friend Node_handle mean(Computation_graph &rec, Node_handle A);
    friend Node_handle mean(Computation_graph &rec, Node_handle A, int dim,
                            bool keep_rank);
    friend Node_handle outer(Computation_graph &rec, Node_handle A,
                             Node_handle B);
    friend Node_handle slice(Computation_graph &rec, Node_handle A,
                             std::initializer_list<int> size,
                             std::initializer_list<int> offset);
    friend Node_handle sum(Computation_graph &rec, Node_handle A);
    friend Node_handle sum(Computation_graph &rec, Node_handle A, int dim,
                           bool keep_rank);
    friend Node_handle transpose(Computation_graph &rec, Node_handle A,
                                 std::initializer_list<int> perm);
    template <class Kernel>
    friend Node_handle unOperator(Computation_graph &rec, Node_handle A);
};

} // namespace kaad
