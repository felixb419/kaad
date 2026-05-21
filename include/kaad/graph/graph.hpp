#pragma once

#include "kaad/graph/internal/inode.hpp"
#include "kaad/graph/operators/internal/kernels.hpp"
#include "kaad/tensor/internal/tensor_types.hpp"

#include <cstddef>
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

    /**
     * @brief Returns a pointer to a node handle.
     * @throws kaad::ArgumentError if @p node doesnt refer to correct graph.
     * @param node Node handle of the relevant node.
     * @return Pointer to the Node.
     */
    [[nodiscard]] INode *get_node(Node node);

    /// @brief Allocate memory for the value and gradient tensor for all nodes.
    void allocate();

    friend class Node;

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

    /// @defgroup operators Operators applied to Nodes on a computation graph.

    /// @defgroup input_operators Operators that add input nodes to the graph.
    /// @ingroup operators

    /**
     * @brief Adds an input node to the computation graph.
     * @ingroup input_operators
     *
     * @param shape Shape of the node.
     * @return A handle of the new input node.
     */
    Node input(ShapeView shape);

    /**
     * @brief Adds multiple input nodes to the computation graph.
     * @ingroup input_operators
     *
     * @param shapes Shapes of the nodes.
     * @return A vector containing the respecitve node handles.
     */
    std::vector<Node> input(std::span<ShapeView> shapes);

  private:
    template <operations::kernels::Binary Kernel>
    Node binary_operator(Node lhs, Node rhs);

  public:
    /// @defgroup binary_operators Operators that take two inputs.
    /// @ingroup operators

    /**
     * @brief Adds a binary addition node to the computation graph.
     * @ingroup binary_operators
     *
     * Computes the element-wise sum of two input tensor nodes @p lhs and @p
     * rhs.
     * @pre Both tensors must have the same shape or be broadcast-compatible
     * (@ref broadcasting).
     *
     * @param lhs Handle of the first input node.
     * @param rhs Handle of the second input node.
     * @return A handle of the new node representing the element-wise sum of @p
     * lhs and @p rhs.
     */
    Node add(Node lhs, Node rhs);

    /**
     * @brief Adds a binary subtratction node to the computation graph.
     * @ingroup binary_operators
     *
     * Computes the element-wise difference of two input tensor nodes @p lhs and
     * @p rhs.
     * @pre Both tensors must have the same shape or be broadcast-compatible
     * (@ref broadcasting).
     *
     * @param lhs Handle of the first input node.
     * @param rhs Handle of the second input node.
     * @return A handle of the new node representing the element-wise difference
     * of
     * @p lhs and @p rhs.
     */
    Node sub(Node lhs, Node rhs);

    /**
     * @brief Adds a binary multiplication node to the computation
     * graph.
     * @ingroup binary_operators
     *
     * Computes the element-wise product of two input tensor nodes @p lhs and
     * @p rhs.
     * @pre Both tensors must have the same shape or be broadcast-compatible
     * (@ref broadcasting).
     *
     * @param lhs Handle of the first input node.
     * @param rhs Handle of the second input node.
     * @return A handle of the new node representing the element-wise product of
     * @p lhs and @p rhs.
     */
    Node mul(Node lhs, Node rhs);

    /**
     * @brief Adds a binary division node to the computation graph.
     * @ingroup binary_operators
     *
     * Computes the element-wise quotient of two input tensor nodes @p lhs and
     * @p rhs.
     * @pre Both tensors must have the same shape or be broadcast-compatible
     * (@ref broadcasting).
     *
     * @param lhs Handle of the first input node.
     * @param rhs Handle of the second input node.
     * @return A handle of the new node representing the element-wise quotient
     * of
     * @p lhs and @p rhs.
     */
    Node div(Node lhs, Node rhs);

    /**
     * @brief Adds a binary power node to the computation graph.
     * @ingroup binary_operators
     *
     * Computes the element-wise power of two input tensor nodes @p lhs and
     * @p rhs.
     * @pre Both tensors must have the same shape or be broadcast-compatible
     * (@ref broadcasting).
     *
     * @param lhs Handle of the first input node.
     * @param rhs Handle of the second input node.
     * @return A handle of the new node representing the element-wise power of
     * @p lhs and @p rhs.
     */
    Node pow(Node lhs, Node rhs);

    /**
     * @brief Adds a binary minimum node to the computation graph.
     * @ingroup binary_operators
     *
     * Computes the element-wise minimum of two input tensor nodes @p lhs and
     * @p rhs.
     * @pre Both tensors must have the same shape or be broadcast-compatible
     * (@ref broadcasting).
     *
     * @param lhs Handle of the first input node.
     * @param rhs Handle of the second input node.
     * @return A handle of the new node representing the element-wise minimum of
     * @p lhs and @p rhs.
     */
    Node min(Node lhs, Node rhs);

    /**
     * @brief Adds a binary maximum node to the computation graph.
     * @ingroup binary_operators
     *
     * Computes the element-wise maximum of two input tensor nodes @p lhs and
     * @p rhs.
     * @pre Both tensors must have the same shape or be broadcast-compatible
     * (@ref broadcasting).
     *
     * @param lhs Handle of the first input node.
     * @param rhs Handle of the second input node.
     * @return A handle of the new node representing the element-wise maximum of
     * @p lhs and @p rhs.
     */
    Node max(Node lhs, Node rhs);

    /**
     * @brief Adds a binary dot product node to the computation graph.
     * @ingroup binary_operators
     *
     * Computes the element-wise dot product of two input tensor nodes @p
     * lhs and
     * @p rhs.
     * @pre Inputs must have rank-1 or rank-0 (scalar).
     *
     * @param lhs Handle of the first input node.
     * @param rhs Handle of the second input node.
     * @return A handle of the new node representing the element-wise dot
     * product of @p lhs and @p rhs.
     */
    Node dot(Node lhs, Node rhs);

    /**
     * @brief Adds a matrix multiplication node to the computation graph.
     * @ingroup binary_operators
     *
     * This operator performs batched matrix multiplication.
     * The shape semantics are split into two parts:
     *
     * 1. **Matrix dimensions (last two axes)**
     *    The last two axes of each input tensor are interpreted as matrices:
     *    - lhs: (..., M, K)
     *    - rhs: (..., K, N)
     *
     *    These dimensions must satisfy standard matrix multiplication rules:
     *
     *    - The inner dimensions must match (K == K)
     *    - The result matrix shape is (..., M, N)
     *
     * 2. **Batch dimensions (all preceding axes)**
     *    All axes before the last two are treated as batch dimensions.
     *    These batch shapes must be broadcast-compatible
     *    (@ref broadcasting).
     *
     *    Broadcasting is applied elementwise over batch dimensions before
     *    performing independent matrix multiplications for each batch.
     *
     * @pre The last two dimensions of @p lhs and @p rhs must be compatible
     *      for matrix multiplication.
     *
     * @pre The batch dimensions (all axes except the last two) must be
     *      broadcast-compatible (@ref broadcasting)
     *      between @p lhs and @p rhs.
     *
     *
     * @param lhs Handle of the first input node.
     * @param rhs Handle of the second input node.
     * @return A handle of the new node representing the batched matrix product
     *         of @p lhs and @p rhs.
     */
    Node matmul(Node lhs, Node rhs);

    /**
     * @brief Adds a generalized outer product node to the computation
     * graph.
     * @ingroup binary_operators
     *
     * Computes the outer product of two input tensor nodes @p lhs and @p
     * rhs. The result is a tensor whose shape is the concatenation of the
     * shapes of
     * @p lhs and @p rhs. For example:
     * - If @p lhs has shape (m,) and @p rhs has shape (n,), the result has
     * shape (m, n).
     * - If @p lhs has shape (m, k) and @p rhs has shape (n, p), the result
     * has shape (m, k, n, p).
     *
     * Each element of the output is computed as the product of an element
     * from
     * @p lhs and an element from @p rhs, preserving the full structure of
     * both input tensors.
     *
     * @param lhs Handle of the first input node.
     * @param rhs Handle of the second input node.
     * @return A Handle of the new node representing the generalized outer
     * product of @p lhs and @p rhs.
     */
    Node outer(Node lhs, Node rhs);

  private:
    template <operations::kernels::Unary Kernel>
    Node unary_operator(Node input);

  public:
    /// @defgroup unary_operators Operators that take one input.
    /// @ingroup operators

    /**
     * @brief Adds a unary negation node to the computation graph.
     * @ingroup unary_operators
     *
     * @param input Handle of the input node.
     * @return A handle of the new node representing the negated tensor,
     *         with the same shape as @p input
     */
    Node negative(Node input);

    /**
     * @brief Adds a unary square node to the computation graph.
     * @ingroup unary_operators
     *
     * @param input Handle of the input node.
     * @return A handle of the new node representing the element-wise square
     * of
     * @p input, with the same shape as the input tensor.
     */
    Node square(Node input);

    /**
     * @brief Adds a unary square root node to the computation graph.
     * @ingroup unary_operators
     *
     * @param input Handle of the input node.
     * @return A handle of the new node representing the element-wise square
     * root of @p input, with the same shape as the input tensor.
     */
    Node sqrt(Node input);

    /**
     * @brief Adds a unary logarithm node to the computation graph.
     * @ingroup unary_operators
     *
     * @param input Handle of the input node.
     * @return A handle of the new node representing the element-wise
     * logarithm of @p input, with the same shape as the input tensor.
     */
    Node log(Node input);

    /**
     * @brief Adds a unary exponent node to the computation graph.
     * @ingroup unary_operators
     *
     * @param input Handle of the input node.
     * @return A handle of the new node representing the element-wise
     * exponent of @p input, with the same shape as the input tensor.
     */
    Node exp(Node input);

    /**
     * @brief Adds a unary absolute value node to the computation graph.
     * @ingroup unary_operators
     *
     * @param input Handle of the input node.
     * @return A handle of the new node representing the element-wise
     * absolute value of @p input, with the same shape as the input tensor.
     */
    Node abs(Node input);

    /**
     * @brief Adds a slice node to the computation graph.
     *
     * Extracts a slice from the input tensor node @p input, starting at the
     * specified @p start and with with a shape specified by @p shape.
     *
     * @ingroup unary_operators
     * @pre @p shape and @p start need to have the same size.
     * @pre @p shape[i] + @p start[i] <= input_shape[i]
     *
     * @param input Handle of the input node.
     * @param shape The shape of the slice.
     * @param start The offset of the slice, will be padded with zeros on
     * the right.
     * @return A handle of the new node representing the slice.
     */
    Node slice(Node input, Shape shape, StaticVector<std::size_t> start = {});

    /**
     * @brief Adds a unary sum node to the computation graph.
     * @ingroup unary_operators
     *
     * Computes the sum of all elements in the input tensor node @p input,
     * producing a scalar tensor node containing the total sum.
     *
     * @param input Handle of the input node.
     * @return A handle of the new node representing the scalar sum of all
     * elements of @p input.
     */
    Node sum(Node input);

    /**
     * @brief Adds a sum node to the computation graph that sums elements
     * along a specified axis.
     * @ingroup unary_operators
     *
     * Computes the sum of elements in the input tensor node @p input along
     * the given @p axis. The resulting tensor shape depends on the @p
     * keep_rank flag:
     * - If @p keep_rank is false (default), the @p axis is removed from
     * the output shape.
     * - If @p keep_rank is true, the @p axis is retained with an extent
     * of 1.
     *
     * @param input Handle of the input node.
     * @param axis The axis along which to sum.
     * @param keep_rank If true, retains the summed axis with an extent of
     * 1; if false, removes it.
     * @return A handle of the new node representing the tensor after
     * summation along the specified axis.
     */
    Node sum(Node input, std::size_t axis, bool keep_rank = false);

    /**
     * @brief Adds a unary mean node to the computation graph.
     * @ingroup unary_operators
     *
     * Computes the mean (average) of all elements in the input tensor node
     * @p input, producing a scalar tensor node containing the total
     * average.
     *
     * @param input Handle of the input node.
     * @return A handle of the new node representing the scalar mean of all
     * elements of @p input.
     */
    Node mean(Node input);

    /**
     * @brief Adds a mean node to the computation graph that computes the
     * mean along a specified axis.
     * @ingroup unary_operators
     *
     * Computes the mean of elements in the input tensor node @p input along
     * the given @p axis. The resulting tensor shape depends on the @p
     * keep_rank flag:
     * - If @p keep_rank is false (default), @p axis is removed from
     * the output shape.
     * - If @p keep_rank is true, the axis @p axis is retained with an
     * extent of 1.
     *
     * @param input Handle of the input node.
     * @param axis The axis along which to compute the mean.
     * @param keep_rank If true, retains the mean-reduced axis with an
     * extent of 1; if false, removes it.
     * @return A handle of the new node representing the tensor after mean
     * reduction along the specified axis.
     */
    Node mean(Node input, std::size_t axis, bool keep_rank = false);

    /**
     * @brief Adds a unary transpose node to the computation graph.
     * @ingroup unary_operators
     *
     * Transposes the input tensor node @p input according to the
     * permutation @p perm. If @p perm is empty, the tensor is fully
     * transposed by reversing its axes.
     *
     * @pre perm must contain a valid permutation of the axes of @p input.
     *
     * @param input Handle of the input node.
     * @param perm An optional initializer list specifying the permutation
     * of axes. If not provided or empty, a full transpose (reverse of all
     * axes) is performed.
     * @return A handle of the new node representing the transposed tensor,
     * with shape adjusted according to @p perm or full transpose.
     */
    Node transpose(Node input, StaticVector<std::size_t> perm = {});
};

} // namespace kaad
