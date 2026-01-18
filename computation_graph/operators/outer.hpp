#pragma once

#include "../../tensor/tensor.hpp"       // for Tensor
#include "../../tensorfuncs/kernels.hpp" // for Kernels
#include "../../tensorfuncs/strides.hpp" // for outer
#include <memory>                        // for std::make_unique

namespace kaad {

template <typename T> struct Computation_graph;
template <typename T> struct INode;
template <typename T, class Kernel> struct Node_binary_flex;

/**
 * @brief Adds a generalized outer product node to the computation graph.
 *
 * Computes the outer product of two input tensor nodes `A_ptr` and `B_ptr`.
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
 * @param A_ptr Pointer to the first input tensor node A.
 * @param B_ptr Pointer to the second input tensor node B.
 * @return A pointer to the new node representing the generalized outer product
 * of A and B.
 */
template <typename T>
INode<T> *outer(Computation_graph<T> &rec, INode<T> *A_ptr, INode<T> *B_ptr) {
    int recLen = rec.nodes.size();
    Tensor<T> &A = A_ptr->value;
    Tensor<T> &B = B_ptr->value;

    size_t newLen = A.nDims() + B.nDims();
    std::vector<int> newShape(newLen);
    std::copy(A.shape_begin(), A.shape_end(), newShape.begin());
    std::copy(B.shape_begin(), B.shape_end(), newShape.begin() + A.nDims());

    using Kernel = typename Kernels::Mul<T>;

    rec.nodes.push_back(std::move(std::make_unique<Node_binary_flex<T, Kernel>>(
        A_ptr, B_ptr, newShape, newLen)));
    Strides::outer<T>(
        *rec.nodes.back().get()); // override strides from constructor

    return rec.nodes.back().get();
}

} // namespace kaad
