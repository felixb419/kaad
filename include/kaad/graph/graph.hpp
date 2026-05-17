#pragma once

#include <cstddef>
#include <kaad/graph/internal/inode.hpp>
#include <kaad/operators/internal/kernels.hpp>
#include <kaad/tensor/internal/tensor_types.hpp>
#include <kaad/tensor/tensor_view.hpp>
#include <memory>
#include <span>
#include <vector>

namespace kaad {

class Node;
template <typename T> class StaticVector;

/**
 * @brief Represents a computation graph for automatic differentiation.
 *
 * This graph stores nodes derived from the INode interface. It supports
 * evaluating node values and computing gradients through backpropagation.
 */
class Graph {
  private:
    std::vector<std::unique_ptr<INode>>
        nodes; ///< Holds unique pointers pointing to computation nodes

    std::vector<Scalar> tensor_buff; ///< Block of memory for value and gradient
                                     ///< tensors in nodes.

    /// @brief Get the last node in the graph.
    /// @return Handle of the node at the back of the node vector.
    [[nodiscard]] Node back_handle() noexcept;

    /**
     * @brief Returns a pointer to a node handle.
     * @throws kaad::ArgumentError if @p node doesnt refer to correct graph.
     * @param node Node handle of the relevant node.
     * @return Pointer to the Node.
     */
    [[nodiscard]] INode *get_node(Node node);

  public:
    /// @brief Allocate memory for the value and gradient tensor for all nodes.
    void allocate();

    /**
     * @brief Constructs an evaluated node with no inputs and adds it to the
     * graph.
     *
     * Creates a Tensor using the given shape, wraps it in a InputNode, and
     * stores the node as a std::unique_ptr in the `nodes` container.
     *
     * @param value_shape Shape of the tensor of the input node.
     * @return A handle of the newly created InputNode.
     */
    [[nodiscard]] Node add_input_node(ShapeView value_shape);

    /**
     * @brief Resets all node values and gradients in the computation graph.
     *
     * Calls the `reset()` method on each node in the graph, zeroing out their
     * associated value and gradient tensors. This is typically used before a
     * new forward pass.
     */
    void reset();

    /**
     * @brief Evaluates a list of nodes and returns their tensor values.
     * @warning reset has to be called before nodes are evaluated.
     *
     * Accepts a variadic list of node handles, evaluates each one by calling
     * its `evaluate()` method, and returns a std::vector of pointers to their
     * resulting Tensor values in the same order.
     *
     * @param nodes A list of handles of nodes to be evaluated.
     * @return An array of tensor views, each corresponding to the value
     * of the evaluated node.
     */
    std::vector<TensorView> evaluate(std::span<const Node> nodes);

    /**
     * @brief Computes gradients of the computation graph with respect to the
     * given input nodes.
     * @warning evaluate has to be called on @p output before this.
     * @warning get_gradient should only be called once before resetting and
     * evaluating again.
     *
     * Initializes the gradient of the output node `output` to 1 and performs
     * backpropagation through the graph. Returns a list of pointers to the
     * gradient tensors corresponding to each input node in @p inputs.
     *
     * @param output Handle of the output node (target function) with respect
     * to which gradients are computed.
     * @param inputs A list of input node handles for which the gradients are
     * requested.
     * @return An array of tensor views representing the gradients
     * df/dx for each input node.
     */
    std::vector<TensorView> get_gradient(Node output,
                                         std::span<const Node> inputs);

    friend class Node;

    template <operations::kernels::Binary Kernel>
    friend Node binary_operator(Graph &rec, Node lhs, Node rhs,
                                const char *opName);
    template <operations::kernels::Unary Kernel>
    friend Node unary_operator(Graph &rec, Node input);
    friend Node dot(Graph &rec, Node lhs, Node rhs);
    friend Node matmul(Graph &rec, Node lhs, Node rhs);
    friend Node mean(Graph &rec, Node input);
    friend Node mean(Graph &rec, Node input, std::size_t axis, bool keep_rank);
    friend Node outer(Graph &rec, Node lhs, Node rhs);
    friend Node slice(Graph &rec, Node input, Shape shape,
                      StaticVector<std::size_t> start);
    friend Node sum(Graph &rec, Node input);
    friend Node sum(Graph &rec, Node input, std::size_t axis, bool keep_rank);
    friend Node transpose(Graph &rec, Node input,
                          StaticVector<std::size_t> perm);
};

} // namespace kaad
