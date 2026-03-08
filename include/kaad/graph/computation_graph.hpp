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

class Node;

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
    Node back_handle() noexcept;

    /**
     * @brief Returns a pointer to a node handle.
     * @warning Will throw argument_error if @p node doesnt refer to
     * correct graph.
     * @param node Node handle of the relevant node.
     * @return Pointer to the Node.
     */
    INode *get_node(Node node);

  public:
    /**
     * @brief Constructs an evaluated node with no inputs and adds it to the
     * graph.
     *
     * Creates a Tensor using the given shape, wraps it in a InputNode, and
     * stores the node as a std::unique_ptr in the `nodes` container.
     *
     * @param[in] value_shape Shape of the tensor of the input node.
     * @return A handle of the newly created InputNode.
     */
    Node add_input_node(std::span<const int> value_shape);

    /**
     * @brief Evaluates a list of nodes and returns their tensor values.
     * @warning reset has to be called before nodes are evaluated.
     *
     * Accepts a variadic list of node handles, evaluates each one by calling
     * its `eval()` method, and returns a std::array of pointers to their
     * resulting Tensor values in the same order.
     *
     * @tparam Nodes Variadic template parameter pack of Node
     * types.
     *
     * @param nodes A list of handles of nodes to be evaluated.
     * @return An array of Tensor* pointers, each corresponding to the value
     * of the evaluated node.
     */
    std::vector<const Tensor *> evaluate(std::span<const Node> nodes);

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
     * @tparam ptrs Variadic template parameter pack of Node types.
     *
     * @param output Handle of the output node (target function) with respect
     * to which gradients are computed.
     * @param wrt A list of input node handles for which the gradients are
     * requested.
     * @return An array of Tensor* pointers representing the gradients df/dx
     * for each input node.
     */
    std::vector<const Tensor *> getGradient(Node output,
                                            std::span<const Node> inputs);

    /**
     * @brief Resets all node values and gradients in the computation graph.
     *
     * Calls the `reset()` method on each node in the graph, zeroing out their
     * associated value and gradient tensors. This is typically used before a
     * new forward pass.
     */
    void reset();

    friend std::ostream &operator<<(std::ostream &os, Node node);

    friend class Node;

    template <class Kernel>
    friend Node binOperator(Computation_graph &rec, Node A, Node B,
                            const char *opName);
    friend Node dot(Computation_graph &rec, Node A, Node B);
    friend Node matmul(Computation_graph &rec, Node A, Node B);
    friend Node mean(Computation_graph &rec, Node A);
    friend Node mean(Computation_graph &rec, Node A, int dim, bool keep_rank);
    friend Node outer(Computation_graph &rec, Node A, Node B);
    friend Node slice(Computation_graph &rec, Node A,
                      std::initializer_list<int> size,
                      std::initializer_list<int> offset);
    friend Node sum(Computation_graph &rec, Node A);
    friend Node sum(Computation_graph &rec, Node A, int dim, bool keep_rank);
    friend Node transpose(Computation_graph &rec, Node A,
                          std::initializer_list<int> perm);
    template <class Kernel>
    friend Node unOperator(Computation_graph &rec, Node A);
};

} // namespace kaad
