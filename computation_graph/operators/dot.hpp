#pragma once

#include "../../tensor/tensor.hpp"           // for Tensor
#include "../../tensorfuncs/adjoint_ops.hpp" // for tensorfuncs::adjoint
#include "../../tensorfuncs/kernels.hpp"     // for Kernels
#include "../../tensorfuncs/primal_ops.hpp"  // for tensorfuncs::primal
#include "exceptions.hpp"                    // for shape_error
#include <memory>                            // for std::make_unique

namespace kaad {

template <typename T> struct Computation_graph;
template <typename T> struct INode;
template <typename T, class Kernel> struct Node_binary;

/**
 * @brief Adds a binary dot product node (A ⋅ B) to the computation graph.
 *
 * Computes the element-wise dot product of two input tensor nodes `A_ptr` and
 * `B_ptr`. Both tensors must have the same shape or be broadcast-compatible.
 *
 * @tparam T The data type of the tensor values.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A_ptr Pointer to the first input tensor node A.
 * @param B_ptr Pointer to the second input tensor node B.
 * @return A pointer to the new node representing the element-wise dot product
 * of A and B.
 */
template <typename T>
INode<T> *dot(Computation_graph<T> &rec, INode<T> *A_ptr, INode<T> *B_ptr) {
    using Op = class Kernels::Null::Op;
    using Grad = class Kernels::Null::Grad;
    tensorfuncs::primal::binary::pointwise_fn<T, Op> scalar =
        tensorfuncs::primal::binary::scalarDot<T, Op>;
    tensorfuncs::adjoint::binary::pointwise_fn<T, Grad> scalar_grad =
        tensorfuncs::adjoint::binary::scalarDot<T, Grad>;
    tensorfuncs::primal::binary::pointwise_fn<T, Op> dot =
        tensorfuncs::primal::binary::dot<T, Op>;
    tensorfuncs::adjoint::binary::pointwise_fn<T, Grad> dot_grad =
        tensorfuncs::adjoint::binary::dot<T, Grad>;

    int recLen = rec.nodes.size();
    Tensor<T> &A = A_ptr->value;
    Tensor<T> &B = B_ptr->value;

    bool A_scalar = A.nDims() == 1 && A.shape[0] == 1;
    bool B_scalar = B.nDims() == 1 && B.shape[0] == 1;
    if (B_scalar) {
        auto newNode = std::make_unique<Node_binary<T, Kernels::Null>>(
            scalar, scalar_grad, A_ptr, B_ptr, ((T)0));
        auto raw_ptr = newNode.get();
        raw_ptr->end = A.data() + A.val.size();
        rec.nodes.push_back(std::move(newNode));
    } else if (A_scalar) {
        auto newNode = std::make_unique<Node_binary<T, Kernels::Null>>(
            scalar, scalar_grad, B_ptr, A_ptr, ((T)0));
        auto raw_ptr = newNode.get();
        raw_ptr->end = B.data() + B.val.size();
        rec.nodes.push_back(std::move(newNode));
    } else if (A.nDims() == 1 && B.nDims() == 1 &&
               std::equal(A.shape.begin(), A.shape.end(), B.shape.begin())) {
        auto newNode = std::make_unique<Node_binary<T, Kernels::Null>>(
            dot, dot_grad, A_ptr, B_ptr, ((T)0));
        auto raw_ptr = newNode.get();
        raw_ptr->end = A.data() + A.val.size();
        rec.nodes.push_back(std::move(newNode));

    } else {
        throw shape_error(recLen, "dot",
                          "incompatible tensor shapes for dot product",
                          {{"A.shape", A.shape}, {"B.shape", B.shape}});
    }

    return rec.nodes.back().get();
}

} // namespace kaad
