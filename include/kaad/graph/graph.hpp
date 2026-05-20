#pragma once

#include <cstddef>
#include <kaad/graph/internal/inode.hpp>
#include <kaad/operators/internal/kernels.hpp>
#include <kaad/tensor/internal/tensor_types.hpp>
#include <kaad/tensor/tensor_view.hpp>
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
    std::vector<INode *>
        nodes; ///< Holds pointers pointing to computation nodes

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
    ~Graph() {
        for (INode *ptr : this->nodes) {
            delete ptr;
        }
    }

    /**
     * @brief Initializes the graph for execution.
     *
     * Must be called before using reset(), evaluate(), or get_gradient().
     */
    void init();

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

    friend Node input(Graph &graph, ShapeView shape);
    template <operations::kernels::Binary Kernel>
    friend Node binary_operator(Graph &graph, Node lhs, Node rhs,
                                const char *opName);
    template <operations::kernels::Unary Kernel>
    friend Node unary_operator(Graph &graph, Node input);
    friend Node dot(Graph &graph, Node lhs, Node rhs);
    friend Node matmul(Graph &graph, Node lhs, Node rhs);
    friend Node mean(Graph &graph, Node input);
    friend Node mean(Graph &graph, Node input, std::size_t axis,
                     bool keep_rank);
    friend Node outer(Graph &graph, Node lhs, Node rhs);
    friend Node slice(Graph &graph, Node input, Shape shape,
                      StaticVector<std::size_t> start);
    friend Node sum(Graph &graph, Node input);
    friend Node sum(Graph &graph, Node input, std::size_t axis, bool keep_rank);
    friend Node transpose(Graph &graph, Node input,
                          StaticVector<std::size_t> perm);
};

} // namespace kaad
