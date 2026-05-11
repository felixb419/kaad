#pragma once

#include <initializer_list>           // for initializer_list
#include <kaad/graph/node_handle.hpp> // for Node
#include <kaad/static_vector.hpp>     // for StaticVector
#include <kaad/tensor/internal/tensor_types.hpp>

namespace kaad {

class Graph;

/**
 * @defgroup operators Operators applied to Nodes on a computation graph.
 */

/**
 * @defgroup unary_operators Unary operators.
 * @ingroup operators
 */

/**
 * @brief Adds a unary negation node to the computation graph.
 * @ingroup unary_operators
 *
 * @param rec The computation graph to which the node will be added.
 * @param input Handle of the input node.
 * @return A handle of the new node representing the negated tensor,
 *         with the same shape as @p input
 */
Node negative(Graph &rec, Node input);

/**
 * @brief Adds a unary square node to the computation graph.
 * @ingroup unary_operators
 *
 * @param rec The computation graph to which the node will be added.
 * @param input Handle of the input node.
 * @return A handle of the new node representing the element-wise square of
 * @p input, with the same shape as the input tensor.
 */
Node square(Graph &rec, Node input);

/**
 * @brief Adds a unary square root node to the computation graph.
 * @ingroup unary_operators
 *
 * @param rec The computation graph to which the node will be added.
 * @param input Handle of the input node.
 * @return A handle of the new node representing the element-wise square root
 * of @p input, with the same shape as the input tensor.
 */
Node sqrt(Graph &rec, Node input);

/**
 * @brief Adds a unary logarithm node to the computation graph.
 * @ingroup unary_operators
 *
 * @param rec The computation graph to which the node will be added.
 * @param input Handle of the input node.
 * @return A handle of the new node representing the element-wise logarithm
 * of @p input, with the same shape as the input tensor.
 */
Node log(Graph &rec, Node input);

/**
 * @brief Adds a unary exponent node to the computation graph.
 * @ingroup unary_operators
 *
 * @param rec The computation graph to which the node will be added.
 * @param input Handle of the input node.
 * @return A handle of the new node representing the element-wise exponent
 * of @p input, with the same shape as the input tensor.
 */
Node exp(Graph &rec, Node input);

/**
 * @brief Adds a unary absolute value node to the computation graph.
 * @ingroup unary_operators
 *
 * @param rec The computation graph to which the node will be added.
 * @param input Handle of the input node.
 * @return A handle of the new node representing the element-wise absolute
 * value of @p input, with the same shape as the input tensor.
 */
Node abs(Graph &rec, Node input);

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
 * @param rec The computation graph to which the node will be added.
 * @param input Handle of the input node.
 * @param shape The shape of the slice.
 * @param start The offset of the slice, will be padded with zeros on the right.
 * @return A handle of the new node representing the slice.
 */
Node slice(Graph &rec, Node input, Shape shape,
           StaticVector<std::size_t> start = {});

/**
 * @brief Adds a unary sum node to the computation graph.
 * @ingroup unary_operators
 *
 * Computes the sum of all elements in the input tensor node @p input,
 * producing a scalar tensor node containing the total sum.
 *
 * @param rec The computation graph to which the node will be added.
 * @param input Handle of the input node.
 * @return A handle of the new node representing the scalar sum of all elements
 * of @p input.
 */
Node sum(Graph &rec, Node input);

/**
 * @brief Adds a sum node to the computation graph that sums elements along a
 * specified axis.
 * @ingroup unary_operators
 *
 * Computes the sum of elements in the input tensor node @p input along the
 * given @p axis. The resulting tensor shape depends on the @p
 * keep_rank flag:
 * - If @p keep_rank is false (default), the @p axis is removed from
 * the output shape.
 * - If @p keep_rank is true, the @p axis is retained with an extent of 1.
 *
 * @param rec The computation graph to which the node will be added.
 * @param input Handle of the input node.
 * @param axis The axis along which to sum.
 * @param keep_rank If true, retains the summed axis with an extent of 1; if
 * false, removes it.
 * @return A handle of the new node representing the tensor after summation
 * along the specified axis.
 */
Node sum(Graph &rec, Node input, std::size_t axis, bool keep_rank = false);

/**
 * @brief Adds a unary mean node to the computation graph.
 * @ingroup unary_operators
 *
 * Computes the mean (average) of all elements in the input tensor node @p
 * input, producing a scalar tensor node containing the total average.
 *
 * @param rec The computation graph to which the node will be added.
 * @param input Handle of the input node.
 * @return A handle of the new node representing the scalar mean of all
 * elements of @p input.
 */
Node mean(Graph &rec, Node input);

/**
 * @brief Adds a mean node to the computation graph that computes the mean along
 * a specified axis.
 * @ingroup unary_operators
 *
 * Computes the mean of elements in the input tensor node @p input along the
 * given @p axis. The resulting tensor shape depends on the @p
 * keep_rank flag:
 * - If @p keep_rank is false (default), @p axis is removed from
 * the output shape.
 * - If @p keep_rank is true, the axis @p axis is retained with an extent of 1.
 *
 * @param rec The computation graph to which the node will be added.
 * @param input Handle of the input node.
 * @param axis The axis along which to compute the mean.
 * @param keep_rank If true, retains the mean-reduced axis with an extent of 1;
 * if false, removes it.
 * @return A handle of the new node representing the tensor after mean
 * reduction along the specified axis.
 */
Node mean(Graph &rec, Node input, std::size_t axis, bool keep_rank = false);

/**
 * @brief Adds a unary transpose node to the computation graph.
 * @ingroup unary_operators
 *
 * Transposes the input tensor node @p input according to the permutation @p
 * perm. If @p perm is empty, the tensor is fully transposed by reversing its
 * axes.
 *
 * @pre perm must contain a valid permutation of the axes of @p input.
 *
 * @param rec The computation graph to which the node will be added.
 * @param input Handle of the input node.
 * @param perm An optional initializer list specifying the permutation of axes.
 * If not provided or empty, a full transpose (reverse of all axes) is
 * performed.
 * @return A handle of the new node representing the transposed tensor, with
 * shape adjusted according to @p perm or full transpose.
 */
Node transpose(Graph &rec, Node input, StaticVector<std::size_t> perm = {});

/**
 * @defgroup binary_operators Binary operators.
 * @ingroup operators
 */

/**
 * @brief Adds a binary addition node to the computation graph.
 * @ingroup binary_operators
 *
 * Computes the element-wise sum of two input tensor nodes @p lhs and @p rhs.
 * @pre Both tensors must have the same shape or be broadcast-compatible.
 *
 * @param rec The computation graph to which the node will be added.
 * @param lhs Handle of the first input node.
 * @param rhs Handle of the second input node.
 * @return A handle of the new node representing the element-wise sum of @p lhs
 * and @p rhs.
 */
Node add(Graph &rec, Node lhs, Node rhs);

/**
 * @brief Adds a binary subtratction node to the computation graph.
 * @ingroup binary_operators
 *
 * Computes the element-wise difference of two input tensor nodes @p lhs and
 * @p rhs.
 * @pre Both tensors must have the same shape or be broadcast-compatible.
 *
 * @param rec The computation graph to which the node will be added.
 * @param lhs Handle of the first input node.
 * @param rhs Handle of the second input node.
 * @return A handle of the new node representing the element-wise difference of
 * @p lhs and @p rhs.
 */
Node sub(Graph &rec, Node lhs, Node rhs);

/**
 * @brief Adds a binary multiplication node to the computation
 * graph.
 * @ingroup binary_operators
 *
 * Computes the element-wise product of two input tensor nodes @p lhs and
 * @p rhs.
 * @pre Both tensors must have the same shape or be broadcast-compatible.
 *
 * @param rec The computation graph to which the node will be added.
 * @param lhs Handle of the first input node.
 * @param rhs Handle of the second input node.
 * @return A handle of the new node representing the element-wise product of
 * @p lhs and @p rhs.
 */
Node mul(Graph &rec, Node lhs, Node rhs);

/**
 * @brief Adds a binary division node to the computation graph.
 * @ingroup binary_operators
 *
 * Computes the element-wise quotient of two input tensor nodes @p lhs and
 * @p rhs.
 * @pre Both tensors must have the same shape or be broadcast-compatible.
 *
 * @param rec The computation graph to which the node will be added.
 * @param lhs Handle of the first input node.
 * @param rhs Handle of the second input node.
 * @return A handle of the new node representing the element-wise quotient of
 * @p lhs and @p rhs.
 */
Node div(Graph &rec, Node lhs, Node rhs);

/**
 * @brief Adds a binary power node to the computation graph.
 * @ingroup binary_operators
 *
 * Computes the element-wise power of two input tensor nodes @p lhs and
 * @p rhs.
 * @pre Both tensors must have the same shape or be broadcast-compatible.
 *
 * @param rec The computation graph to which the node will be added.
 * @param lhs Handle of the first input node.
 * @param rhs Handle of the second input node.
 * @return A handle of the new node representing the element-wise power of
 * @p lhs and @p rhs.
 */
Node pow(Graph &rec, Node lhs, Node rhs);

/**
 * @brief Adds a binary minimum node to the computation graph.
 * @ingroup binary_operators
 *
 * Computes the element-wise minimum of two input tensor nodes @p lhs and
 * @p rhs.
 * @pre Both tensors must have the same shape or be broadcast-compatible.
 *
 * @param rec The computation graph to which the node will be added.
 * @param lhs Handle of the first input node.
 * @param rhs Handle of the second input node.
 * @return A handle of the new node representing the element-wise minimum of
 * @p lhs and @p rhs.
 */
Node min(Graph &rec, Node lhs, Node rhs);

/**
 * @brief Adds a binary maximum node to the computation graph.
 * @ingroup binary_operators
 *
 * Computes the element-wise maximum of two input tensor nodes @p lhs and
 * @p rhs.
 * @pre Both tensors must have the same shape or be broadcast-compatible.
 *
 * @param rec The computation graph to which the node will be added.
 * @param lhs Handle of the first input node.
 * @param rhs Handle of the second input node.
 * @return A handle of the new node representing the element-wise maximum of
 * @p lhs and @p rhs.
 */
Node max(Graph &rec, Node lhs, Node rhs);

/**
 * @brief Adds a binary dot product node to the computation graph.
 * @ingroup binary_operators
 *
 * Computes the element-wise dot product of two input tensor nodes @p lhs and
 * @p rhs.
 * @pre Inputs must have rank-1 or rank-0 (scalar).
 *
 * @param rec The computation graph to which the node will be added.
 * @param lhs Handle of the first input node.
 * @param rhs Handle of the second input node.
 * @return A handle of the new node representing the element-wise dot product
 * of @p lhs and @p rhs.
 */
Node dot(Graph &rec, Node lhs, Node rhs);

/**
 * @brief Adds a matrix multiplication node to the computation graph.
 * @ingroup binary_operators
 *
 * This operator supports standard matrix multiplication as well as batch
 * matrix multiplication. For rank-2 inputs, it performs a classic matrix
 * multiplication. For higher-rank tensors, the leading axes are treated as
 * batch axes and are broadcast according to standard broadcasting rules before
 * performing the matrix multiplication on the last two axes.
 *
 * @pre @p lhs and @p rhs need to have shapes compatible for matrix
 * multiplication.
 *
 * @param rec The computation graph to which the node will be added.
 * @param lhs Handle of the first input node.
 * @param rhs Handle of the second input node.
 * @return A handle of the new node representing the matrix product of @p lhs
 * and @p rhs.
 */
Node matmul(Graph &rec, Node lhs, Node rhs);

/**
 * @brief Adds a generalized outer product node to the computation graph.
 * @ingroup binary_operators
 *
 * Computes the outer product of two input tensor nodes @p lhs and @p rhs.
 * The result is a tensor whose shape is the concatenation of the shapes of @p
 * lhs and @p rhs. For example:
 * - If @p lhs has shape (m,) and @p rhs has shape (n,), the result has shape
 * (m, n).
 * - If @p lhs has shape (m, k) and @p rhs has shape (n, p), the result has
 * shape (m, k, n, p).
 *
 * Each element of the output is computed as the product of an element from @p
 * lhs and an element from @p rhs, preserving the full structure of both input
 * tensors.
 *
 * @param rec The computation graph to which the node will be added.
 * @param lhs Handle of the first input node.
 * @param rhs Handle of the second input node.
 * @return A Handle of the new node representing the generalized outer product
 * of @p lhs and @p rhs.
 */
Node outer(Graph &rec, Node lhs, Node rhs);

} // namespace kaad
