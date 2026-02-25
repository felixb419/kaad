#pragma once

#include "../node_handle.hpp" // for Node_handle
#include <initializer_list>   // for initializer_list

namespace kaad {

class Computation_graph;

/**
 * @defgroup ops Operators applied to Nodes on a computation graph.
 */

/**
 * @defgroup unary_ops Unary operators.
 * @ingroup ops
 */

/**
 * @brief Adds a unary negation node (-A) to the computation graph.
 * @ingroup unary_ops
 *
 * @param rec The computation graph to which the node will be added.
 * @param A Handle of the input tensor node A.
 * @return A handle of the new node representing the negated tensor,
 *         with the same shape as A.
 */
Node_handle negative(Computation_graph &rec, Node_handle A);

/**
 * @brief Adds a unary square node (A^2) to the computation graph.
 * @ingroup unary_ops
 *
 * @param rec The computation graph to which the node will be added.
 * @param A Handle of the input tensor node A.
 * @return A handle of the new node representing the element-wise square of A,
 *         with the same shape as the input tensor.
 */
Node_handle square(Computation_graph &rec, Node_handle A);

/**
 * @brief Adds a unary square root node (sqrt(A)) to the computation graph.
 * @ingroup unary_ops
 *
 * @param rec The computation graph to which the node will be added.
 * @param A Handle of the input tensor node A.
 * @return A handle of the new node representing the element-wise square root
 * of A, with the same shape as the input tensor.
 */
Node_handle sqrt(Computation_graph &rec, Node_handle A);

/**
 * @brief Adds a unary logarithm node (log(A)) to the computation graph.
 * @ingroup unary_ops
 *
 * @param rec The computation graph to which the node will be added.
 * @param A Handle of the input tensor node A.
 * @return A handle of the new node representing the element-wise logarithm
 * of A, with the same shape as the input tensor.
 */
Node_handle log(Computation_graph &rec, Node_handle A);

/**
 * @brief Adds a unary exponent node (e^A) to the computation graph.
 * @ingroup unary_ops
 *
 * @param rec The computation graph to which the node will be added.
 * @param A Handle of the input tensor node A.
 * @return A handle of the new node representing the element-wise exponent
 * of A, with the same shape as the input tensor.
 */
Node_handle exp(Computation_graph &rec, Node_handle A);

/**
 * @brief Adds a unary absolute value node (|A|) to the computation graph.
 * @ingroup unary_ops
 *
 * @param rec The computation graph to which the node will be added.
 * @param A Handle of the input tensor node A.
 * @return A handle of the new node representing the element-wise absolute
 * value of A, with the same shape as the input tensor.
 */
Node_handle abs(Computation_graph &rec, Node_handle A);

/**
 * @brief Adds a slice node to the computation graph.
 * @ingroup unary_ops
 *
 * Extracts a slice from the input tensor node `A`, starting at the
 * specified `offset` and extending for the given `size` along each dimension.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A Handle of the input tensor node A.
 * @param size An initializer list specifying the size (length) of the slice
 * along each dimension, unspecified dimensions stay the same.
 * @param offset An initializer list specifying the starting indices (offset)
 * for the slice along each dimension, unspecified dimensions are assumed to be
 * 0.
 * @return A handle of the new node representing the sliced tensor.
 */
Node_handle slice(Computation_graph &rec, Node_handle A,
                  std::initializer_list<int> size,
                  std::initializer_list<int> offset);

/**
 * @brief Adds a unary sum node to the computation graph.
 * @ingroup unary_ops
 *
 * Computes the sum of all elements in the input tensor node `A`,
 * producing a scalar tensor node containing the total sum.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A Handle of the input tensor node A.
 * @return A handle of the new node representing the scalar sum of all elements
 * of A.
 */
Node_handle sum(Computation_graph &rec, Node_handle A);

/**
 * @brief Adds a sum node to the computation graph that sums elements along a
 * specified dimension.
 * @ingroup unary_ops
 *
 * Computes the sum of elements in the input tensor node `A` along the given
 * dimension `dim`. The resulting tensor shape depends on the `keepNDims` flag:
 * - If `keepNDims` is false (default), the dimension `dim` is removed from the
 * output shape.
 * - If `keepNDims` is true, the dimension `dim` is retained with size 1.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A Handle of the input tensor node A.
 * @param dim The dimension along which to sum.
 * @param keepNDims If true, retains the summed dimension with size 1; if false,
 * removes it.
 * @return A handle of the new node representing the tensor after summation
 * along the specified dimension.
 */
Node_handle sum(Computation_graph &rec, Node_handle A, int dim,
                bool keepNDims = false);

/**
 * @brief Adds a unary transpose node to the computation graph.
 * @ingroup unary_ops
 *
 * Transposes the input tensor node `A` according to the permutation `perm`.
 * If `perm` is empty, the tensor is fully transposed by reversing its
 * dimensions.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A Handle of the input tensor node A.
 * @param perm An optional initializer list specifying the permutation of axes.
 *             If not provided or empty, a full transpose (reverse of all axes)
 * is performed.
 * @return A handle of the new node representing the transposed tensor,
 *         with shape adjusted according to `perm` or full transpose.
 */
Node_handle transpose(Computation_graph &rec, Node_handle A,
                      std::initializer_list<int> perm = {});

/**
 * @defgroup binary_ops Binary operators.
 * @ingroup ops
 */

/**
 * @brief Adds a binary addition node (A + B) to the computation graph.
 * @ingroup binary_ops
 *
 * Computes the element-wise sum of two input tensor nodes `A` and `B`.
 * Both tensors must have the same shape or be broadcast-compatible.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A Handle of the first input tensor node A.
 * @param B Handle of the second input tensor node B.
 * @return A handle of the new node representing the element-wise sum of A and
 * B.
 */
Node_handle add(Computation_graph &rec, Node_handle A, Node_handle B);

/**
 * @brief Adds a binary subtratction node (A - B) to the computation graph.
 * @ingroup binary_ops
 *
 * Computes the element-wise difference of two input tensor nodes `A` and
 * `B`. Both tensors must have the same shape or be broadcast-compatible.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A Handle of the first input tensor node A.
 * @param B Handle of the second input tensor node B.
 * @return A handle of the new node representing the element-wise difference of
 * A and B.
 */
Node_handle sub(Computation_graph &rec, Node_handle A, Node_handle B);

/**
 * @brief Adds a binary multiplication node (A * B) to the computation
 * graph.
 * @ingroup binary_ops
 *
 * Computes the element-wise product of two input tensor nodes `A` and
 * `B`. Both tensors must have the same shape or be broadcast-compatible.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A Handle of the first input tensor node A.
 * @param B Handle of the second input tensor node B.
 * @return A handle of the new node representing the element-wise product of A
 * and B.
 */
Node_handle mul(Computation_graph &rec, Node_handle A, Node_handle B);

/**
 * @brief Adds a binary division node (A / B) to the computation graph.
 * @ingroup binary_ops
 *
 * Computes the element-wise quotient of two input tensor nodes `A` and
 * `B`. Both tensors must have the same shape or be broadcast-compatible.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A Handle of the first input tensor node A.
 * @param B Handle of the second input tensor node B.
 * @return A handle of the new node representing the element-wise quotient of A
 * and B.
 */
Node_handle div(Computation_graph &rec, Node_handle A, Node_handle B);

/**
 * @brief Adds a binary power node (A ^ B) to the computation graph.
 * @ingroup binary_ops
 *
 * Computes the element-wise power of two input tensor nodes `A` and
 * `B`. Both tensors must have the same shape or be broadcast-compatible.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A Handle of the first input tensor node A.
 * @param B Handle of the second input tensor node B.
 * @return A handle of the new node representing the element-wise power of A
 * and B.
 */
Node_handle pow(Computation_graph &rec, Node_handle A, Node_handle B);

/**
 * @brief Adds a binary minimum node (min(A, B)) to the computation graph.
 * @ingroup binary_ops
 *
 * Computes the element-wise minimum of two input tensor nodes `A` and
 * `B`. Both tensors must have the same shape or be broadcast-compatible.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A Handle of the first input tensor node A.
 * @param B Handle of the second input tensor node B.
 * @return A handle of the new node representing the element-wise minimum of A
 * and B.
 */
Node_handle min(Computation_graph &rec, Node_handle A, Node_handle B);

/**
 * @brief Adds a binary maximum node (max(A, B)) to the computation graph.
 * @ingroup binary_ops
 *
 * Computes the element-wise maximum of two input tensor nodes `A` and
 * `B`. Both tensors must have the same shape or be broadcast-compatible.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A Handle of the first input tensor node A.
 * @param B Handle of the second input tensor node B.
 * @return A handle of the new node representing the element-wise maximum of A
 * and B.
 */
Node_handle max(Computation_graph &rec, Node_handle A, Node_handle B);

/**
 * @brief Adds a binary dot product node (A dot B) to the computation graph.
 * @ingroup binary_ops
 *
 * Computes the element-wise dot product of two input tensor nodes `A` and
 * `B`. Both tensors must have the same shape or be broadcast-compatible.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A Handle of the first input tensor node A.
 * @param B Handle of the second input tensor node B.
 * @return A handle of the new node representing the element-wise dot product
 * of A and B.
 */
Node_handle dot(Computation_graph &rec, Node_handle A, Node_handle B);

/**
 * @brief Adds a matrix multiplication node (A x B) to the computation graph.
 * @ingroup binary_ops
 *
 * Performs matrix multiplication between two input tensor nodes `A` and
 * `B`. Supports both standard 2D matrix multiplication and batched matrix
 * multiplication:
 * - If both tensors are 2D, performs standard matrix multiplication.
 * - If tensors have more than 2 dimensions, performs batched matrix
 * multiplication over the leading dimensions. For example, multiplying tensors
 * of shape (batch, M, K) x (batch, K, N) yields a result of shape (batch, M,
 * N).
 *
 * @param rec The computation graph to which the node will be added.
 * @param A Handle of the left-hand-side input tensor node A.
 * @param B Handle of the right-hand-side input tensor node B.
 * @return A handle of the new node representing the matrix (or batched)
 * product of A and B.
 */
Node_handle matmul(Computation_graph &rec, Node_handle A, Node_handle B);

/**
 * @brief Adds a unary mean node to the computation graph.
 * @ingroup binary_ops
 *
 * Computes the mean (average) of all elements in the input tensor node `A`,
 * producing a scalar tensor node containing the total average.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A Handle of the input tensor node A.
 * @return A handle of the new node representing the scalar mean of all
 * elements of A.
 */
Node_handle mean(Computation_graph &rec, Node_handle A);

/**
 * @brief Adds a mean node to the computation graph that computes the mean along
 * a specified dimension.
 * @ingroup binary_ops
 *
 * Computes the mean of elements in the input tensor node `A` along the
 * given dimension `dim`. The resulting tensor shape depends on the `keepNDims`
 * flag:
 * - If `keepNDims` is false (default), the dimension `dim` is removed from the
 * output shape.
 * - If `keepNDims` is true, the dimension `dim` is retained with size 1.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A Handle of the input tensor node A.
 * @param dim The dimension along which to compute the mean.
 * @param keepNDims If true, retains the mean-reduced dimension with size 1; if
 * false, removes it.
 * @return A handle of the new node representing the tensor after mean
 * reduction along the specified dimension.
 */
Node_handle mean(Computation_graph &rec, Node_handle A, int dim,
                 bool keepNDims = 0);

/**
 * @brief Adds a generalized outer product node to the computation graph.
 * @ingroup binary_ops
 *
 * Computes the outer product of two input tensor nodes `A` and `B`.
 * The result is a tensor whose shape is the concatenation of the shapes of A
 * and B. For example:
 * - If A has shape (m,) and B has shape (n,), the result has shape (m, n).
 * - If A has shape (m, k) and B has shape (n, p), the result has shape (m, k,
 * n, p).
 *
 * Each element of the output is computed as the product of an element from A
 * and an element from B, preserving the full structure of both input tensors.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A Handle of the first input tensor node A.
 * @param B Handle of the second input tensor node B.
 * @return A Handle of the new node representing the generalized outer product
 * of A and B.
 */
Node_handle outer(Computation_graph &rec, Node_handle A, Node_handle B);

} // namespace kaad
