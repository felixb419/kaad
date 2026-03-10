#pragma once

#include "../graph/node_handle.hpp" // for Node
#include <initializer_list>         // for initializer_list

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
Node negative(Graph &rec, Node inpuinput);

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
 * @ingroup unary_operators
 *
 * Extracts a slice from the input tensor node `input`, starting at the
 * specified `offset` and extending for the given `size` along each dimension.
 *
 * @param rec The computation graph to which the node will be added.
 * @param input Handle of the input node.
 * @param size An initializer list specifying the size (length) of the slice
 * along each dimension, unspecified dimensions stay the same.
 * @param offset An initializer list specifying the starting indices (offset)
 * for the slice along each dimension, unspecified dimensions are assumed to be
 * 0.
 * @return A handle of the new node representing the sliced tensor.
 */
Node slice(Graph &rec, Node input, std::initializer_list<int> size,
           std::initializer_list<int> offset);

/**
 * @brief Adds a unary sum node to the computation graph.
 * @ingroup unary_operators
 *
 * Computes the sum of all elements in the input tensor node `A`,
 * producing a scalar tensor node containing the total sum.
 *
 * @param rec The computation graph to which the node will be added.
 * @param input Handle of the input node.
 * @return A handle of the new node representing the scalar sum of all elements
 * of A.
 */
Node sum(Graph &rec, Node input);

/**
 * @brief Adds a sum node to the computation graph that sums elements along a
 * specified dimension.
 * @ingroup unary_operators
 *
 * Computes the sum of elements in the input tensor node `A` along the given
 * dimension `dim`. The resulting tensor shape depends on the `keep_rank` flag:
 * - If `keep_rank` is false (default), the dimension `dim` is removed from the
 * output shape.
 * - If `keep_rank` is true, the dimension `dim` is retained with size 1.
 *
 * @param rec The computation graph to which the node will be added.
 * @param node Handle of the input node.
 * @param dim The dimension along which to sum.
 * @param keep_rank If true, retains the summed dimension with size 1; if false,
 * removes it.
 * @return A handle of the new node representing the tensor after summation
 * along the specified dimension.
 */
Node sum(Graph &rec, Node input, int dim, bool keep_rank = false);

/**
 * @brief Adds a unary mean node to the computation graph.
 * @ingroup unary_operators
 *
 * Computes the mean (average) of all elements in the input tensor node `input`,
 * producing a scalar tensor node containing the total average.
 *
 * @param rec The computation graph to which the node will be added.
 * @param input Handle of the input node.
 * @return A handle of the new node representing the scalar mean of all
 * elements of @p input.
 */
Node mean(Graph &rec, Node input);

/**
 * @brief Adds a mean node to the computation graph that computes the mean along
 * a specified dimension.
 * @ingroup unary_operators
 *
 * Computes the mean of elements in the input tensor node `input` along the
 * given dimension `dim`. The resulting tensor shape depends on the `keep_rank`
 * flag:
 * - If `keep_rank` is false (default), the dimension `dim` is removed from the
 * output shape.
 * - If `keep_rank` is true, the dimension `dim` is retained with size 1.
 *
 * @param rec The computation graph to which the node will be added.
 * @param input Handle of the input node.
 * @param dim The dimension along which to compute the mean.
 * @param keep_rank If true, retains the mean-reduced dimension with size 1; if
 * false, removes it.
 * @return A handle of the new node representing the tensor after mean
 * reduction along the specified dimension.
 */
Node mean(Graph &rec, Node input, int dim, bool keep_rank = 0);

/**
 * @brief Adds a unary transpose node to the computation graph.
 * @ingroup unary_operators
 *
 * Transposes the input tensor node `A` according to the permutation `perm`.
 * If `perm` is empty, the tensor is fully transposed by reversing its
 * dimensions.
 *
 * @param rec The computation graph to which the node will be added.
 * @param input Handle of the input node.
 * @param perm An optional initializer list specifying the permutation of axes.
 *             If not provided or empty, a full transpose (reverse of all axes)
 * is performed.
 * @return A handle of the new node representing the transposed tensor,
 *         with shape adjusted according to `perm` or full transpose.
 */
Node transpose(Graph &rec, Node input, std::initializer_list<int> perm = {});

/**
 * @defgroup binary_operators Binary operators.
 * @ingroup operators
 */

/**
 * @brief Adds a binary addition node to the computation graph.
 * @ingroup binary_operators
 *
 * Computes the element-wise sum of two input tensor nodes `lhs` and `rhs`.
 * Both tensors must have the same shape or be broadcast-compatible.
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
 * Computes the element-wise difference of two input tensor nodes `lhs` and
 * `rhs`. Both tensors must have the same shape or be broadcast-compatible.
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
 * Computes the element-wise product of two input tensor nodes `lhs` and
 * `rhs`. Both tensors must have the same shape or be broadcast-compatible.
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
 * Computes the element-wise quotient of two input tensor nodes `lhs` and
 * `rhs`. Both tensors must have the same shape or be broadcast-compatible.
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
 * Computes the element-wise power of two input tensor nodes `lhs` and
 * `rhs`. Both tensors must have the same shape or be broadcast-compatible.
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
 * Computes the element-wise minimum of two input tensor nodes `lhs` and
 * `rhs`. Both tensors must have the same shape or be broadcast-compatible.
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
 * Computes the element-wise maximum of two input tensor nodes `lhs` and
 * `rhs`. Both tensors must have the same shape or be broadcast-compatible.
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
 * Computes the element-wise dot product of two input tensor nodes `lhs` and
 * `rhs`. Both tensors must have the same shape or be broadcast-compatible.
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
 * Performs matrix multiplication between two input tensor nodes `lhs` and
 * `rhs`. Supports both standard 2D matrix multiplication and batched matrix
 * multiplication:
 * - If both tensors are rank-2, performs standard matrix multiplication.
 * - If tensors have more than 2 dimensions, performs batched matrix
 * multiplication over the leading dimensions. For example, multiplying tensors
 * of shape (batch, M, K) x (batch, K, N) yields a result of shape (batch, M,
 * N).
 *
 * @param rec The computation graph to which the node will be added.
 * @param lhs Handle of the left-hand-side input node.
 * @param rhs Handle of the right-hand-side input node.
 * @return A handle of the new node representing the matrix (or batched)
 * product of @p lhs and @p rhs.
 */
Node matmul(Graph &rec, Node lhs, Node rhs);

/**
 * @brief Adds a generalized outer product node to the computation graph.
 * @ingroup binary_operators
 *
 * Computes the outer product of two input tensor nodes `lhs` and `rhs`.
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
