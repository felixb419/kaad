#pragma once

#include "../../tensor/tensor.hpp" // for Tensor
#include "exceptions.hpp"          // for shape_error
#include <memory>                  // for std::make_unique

namespace kaad {

template <typename T> class Computation_graph;
template <typename T> class INode;
template <typename T> class Node_matmul;
template <typename T> class Node_batch_matmul;

/**
 * @brief Adds a matrix multiplication node (A × B) to the computation graph.
 *
 * Performs matrix multiplication between two input tensor nodes `A_ptr` and
 * `B_ptr`. Supports both standard 2D matrix multiplication and batched matrix
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
 * @param A_ptr Pointer to the left-hand-side input tensor node A.
 * @param B_ptr Pointer to the right-hand-side input tensor node B.
 * @return A pointer to the new node representing the matrix (or batched)
 * product of A and B.
 */
template <typename T>
INode<T> *matmul(Computation_graph<T> &rec, INode<T> *A_ptr, INode<T> *B_ptr) {
    int recLen = rec.nodes.size();
    Tensor<T> &A = A_ptr->value;
    Tensor<T> &B = B_ptr->value;

    size_t newLen = std::max(A.nDims(), B.nDims());
    std::vector<int> newShape(newLen);

    const char *opName = newLen == 2 ? "matmul" : "batch_matmul";
    if (!combine_matrix(A.shape_begin(), A.nDims(), B.shape_begin(), B.nDims(),
                        newShape.data(), newLen)) {
        throw shape_error(
            recLen, opName,
            "incompatible tensor shapes for matrix multiplication",
            {{"A.shape", A.shape()}, {"B.shape", B.shape()}});
    }

    if (newLen == 2) {
        rec.nodes.push_back(std::move(
            std::make_unique<Node_matmul<T>>(A_ptr, B_ptr, newShape)));
    } else {
        rec.nodes.push_back(std::move(
            std::make_unique<Node_batch_matmul<T>>(A_ptr, B_ptr, newShape)));
    }

    return rec.nodes.back().get();
}

} // namespace kaad
