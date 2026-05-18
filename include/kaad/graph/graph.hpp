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

    /// @brief Allocate memory for the value and gradient tensor for all nodes.
    void allocate();

  public:
    /**
     * @brief Initializes the graph for execution.
     *
     * Must be called before using reset(), evaluate(), or get_gradient().
     */
    void init();

    /**
     * @brief Constructs an evaluated node with no inputs and adds it to the
     * graph.
     *
     * Creates a Tensor using the given shape, wraps it in an InputNode, and
     * stores the node in the graph.
     *
     * @param value_shape Shape of the tensor of the input node.
     * @return A handle to the newly created InputNode.
     */
    [[nodiscard]] Node add_input_node(ShapeView value_shape);

    /**
     * @brief Resets all node values and gradients in the graph.
     *
     * Must be called after init() and before each new call to evaluate().
     * Clears all cached values and accumulated gradients.
     */
    void reset();

    /**
     * @brief Evaluates a list of nodes and returns their tensor values.
     *
     * Requires init() and a preceding call to reset().
     * May only be called once per reset().
     *
     * @param nodes Handles of the nodes to evaluate.
     * @return Tensor views of the evaluated node values, in the same order.
     */
    std::vector<TensorView> evaluate(std::span<const Node> nodes);

    /**
     * @brief Computes gradients with respect to the given input nodes.
     *
     * Requires init() and a preceding call to evaluate() on @p output.
     * May only be called once per reset(); call reset() and evaluate() again
     * before requesting another gradient.
     *
     * @param output Handle of the output node (target function).
     * @param inputs Handles of the input nodes for which gradients are
     * requested.
     * @return Tensor views representing df/dx for each input node.
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
