#pragma once

#include "../../tensor/tensor.hpp"   // for Tensor
#include "../common.hpp"             // for combine_matrix
#include "../computation_graph.hpp"  // for Computation_graph
#include "../node_handle.hpp"        // for Node_handle
#include "../nodes/batch_matmul.hpp" // for Node_batch_matmul
#include "../nodes/matmul.hpp"       // for Node_matmul
#include "exceptions.hpp"            // for shape_error
#include <memory>                    // for std::make_unique

namespace kaad {

/**
 * @brief Adds a matrix multiplication node (A × B) to the computation graph.
 *
 * Performs matrix multiplication between two input tensor nodes `A` and
 * `B`. Supports both standard 2D matrix multiplication and batched matrix
 * multiplication:
 * - If both tensors are 2D, performs standard matrix multiplication.
 * - If tensors have more than 2 dimensions, performs batched matrix
 * multiplication over the leading dimensions. For example, multiplying tensors
 * of shape (batch, M, K) × (batch, K, N) yields a result of shape (batch, M,
 * N).
 *
 * @tparam T The data type of the tensor values.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A Handle of the left-hand-side input tensor node A.
 * @param B Handle of the right-hand-side input tensor node B.
 * @return A handle of the new node representing the matrix (or batched)
 * product of A and B.
 */
template <typename T>
Node_handle matmul(Computation_graph &rec, Node_handle A, Node_handle B) {
    int recLen = rec.nodes.size();

    INode *A_ptr = rec.get_node(A);
    INode *B_ptr = rec.get_node(B);
    Tensor &A_val = A_ptr->value;
    Tensor &B_val = B_ptr->value;

    size_t newLen = std::max(A_val.nDims(), B_val.nDims());
    std::vector<int> newShape(newLen);

    const char *opName = newLen == 2 ? "matmul" : "batch_matmul";
    if (!detail::combine_matrix(A_val.shape_begin(), A_val.nDims(),
                                B_val.shape_begin(), B_val.nDims(),
                                newShape.data(), newLen)) {
        throw shape_error(
            recLen, opName,
            "incompatible tensor shapes for matrix multiplication",
            {{"A.shape", A_val.shape()}, {"B.shape", B_val.shape()}});
    }

    if (newLen == 2) {
        rec.nodes.push_back(
            std::move(std::make_unique<Node_matmul>(A_ptr, B_ptr, newShape)));
    } else {
        rec.nodes.push_back(std::move(
            std::make_unique<Node_batch_matmul>(A_ptr, B_ptr, newShape)));
    }

    return rec.back_handle();
}

} // namespace kaad
