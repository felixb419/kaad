#pragma once

#include "../../tensor/tensor.hpp"           // for Tensor
#include "../../tensorfuncs/adjoint_ops.hpp" // for tensorfuncs::adjoint
#include "../../tensorfuncs/kernels.hpp"     // for Kernels
#include "../../tensorfuncs/primal_ops.hpp"  // for tensorfuncs::primal
#include "exceptions.hpp"                    // for shape_error
#include <memory>                            // for std::make_unique

namespace kaad {

template <typename T> class Computation_graph;
template <typename T> class INode;
template <typename T> class Node_handle;
template <typename T, class Kernel> class Node_binary;

/**
 * @brief Adds a binary dot product node (A ⋅ B) to the computation graph.
 *
 * Computes the element-wise dot product of two input tensor nodes `A` and
 * `B`. Both tensors must have the same shape or be broadcast-compatible.
 *
 * @tparam T The data type of the tensor values.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A Handle of the first input tensor node A.
 * @param B Handle of the second input tensor node B.
 * @return A handle of the new node representing the element-wise dot product
 * of A and B.
 */
template <typename T>
Node_handle<T> dot(Computation_graph<T> &rec, Node_handle<T> A,
                   Node_handle<T> B) {
    tensorfuncs::primal::binary::pointwise_fn<T, Kernels::Null> scalar =
        tensorfuncs::primal::binary::scalarDot<T>;
    tensorfuncs::adjoint::binary::pointwise_fn<T, Kernels::Null> scalar_grad =
        tensorfuncs::adjoint::binary::scalarDot<T, Kernels::Null>;

    tensorfuncs::primal::binary::pointwise_fn<T, Kernels::Null> dot =
        tensorfuncs::primal::binary::dot<T, Kernels::Null>;
    tensorfuncs::adjoint::binary::pointwise_fn<T, Kernels::Null> dot_grad =
        tensorfuncs::adjoint::binary::dot<T, Kernels::Null>;

    int recLen = rec.nodes.size();

    INode<T> *A_ptr = rec.get_node(A);
    INode<T> *B_ptr = rec.get_node(B);
    Tensor &A_val = A_ptr->value;
    Tensor &B_val = B_ptr->value;

    bool A_scalar = A_val.nDims() == 1 && A_val.shape()[0] == 1;
    bool B_scalar = B_val.nDims() == 1 && B_val.shape()[0] == 1;
    if (B_scalar) {

        rec.nodes.push_back(
            std::move(std::make_unique<Node_binary<T, Kernels::Null>>(
                scalar, scalar_grad, A_ptr, B_ptr, ((T)0))));
        static_cast<Node_binary<T, Kernels::Null> *>(rec.nodes.back().get())
            ->end = A_val.data() + A_val.size();

    } else if (A_scalar) {

        rec.nodes.push_back(
            std::move(std::make_unique<Node_binary<T, Kernels::Null>>(
                scalar, scalar_grad, B_ptr, A_ptr, ((T)0))));
        static_cast<Node_binary<T, Kernels::Null> *>(rec.nodes.back().get())
            ->end =
            B_val.data() + B_val.size(); // override end from constructor

    } else if (A_val.nDims() == 1 && B_val.nDims() == 1 &&
               std::equal(A_val.shape_begin(), A_val.shape_end(),
                          B_val.shape_begin())) {

        rec.nodes.push_back(
            std::move(std::make_unique<Node_binary<T, Kernels::Null>>(
                dot, dot_grad, A_ptr, B_ptr, ((T)0))));
        static_cast<Node_binary<T, Kernels::Null> *>(rec.nodes.back().get())
            ->end = A_val.data() + A_val.size();

    } else {
        throw shape_error(
            recLen, "dot", "incompatible tensor shapes for dot product",
            {{"A.shape", A_val.shape()}, {"B.shape", B_val.shape()}});
    }

    return rec.back_handle();
}

} // namespace kaad
