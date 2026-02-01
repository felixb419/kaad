#pragma once

#include "../../tensor/tensor.hpp"  // for Tensor
#include "../computation_graph.hpp" // for Computation_graph
#include "../node_handle.hpp"       // for Node_handle
#include "../nodes/outer.hpp"       // for Node_outer
#include <memory>                   // for std::make_unique

namespace kaad {

/**
 * @brief Adds a generalized outer product node to the computation graph.
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
 * @tparam T The data type of the tensor values.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A Handle of the first input tensor node A.
 * @param B Handle of the second input tensor node B.
 * @return A Handle of the new node representing the generalized outer product
 * of A and B.
 */
template <typename T>
Node_handle outer(Computation_graph &rec, Node_handle A, Node_handle B) {
    int recLen = rec.nodes.size();

    INode *A_ptr = rec.get_node(A);
    INode *B_ptr = rec.get_node(B);
    Tensor &A_val = A_ptr->value;
    Tensor &B_val = B_ptr->value;

    size_t newLen = A_val.nDims() + B_val.nDims();
    std::vector<int> newShape(newLen);
    std::copy(A_val.shape_begin(), A_val.shape_end(), newShape.begin());
    std::copy(B_val.shape_begin(), B_val.shape_end(),
              newShape.begin() + A_val.nDims());

    rec.nodes.push_back(
        std::move(std::make_unique<Node_outer>(A_ptr, B_ptr, newShape)));

    return rec.back_handle();
}

} // namespace kaad
