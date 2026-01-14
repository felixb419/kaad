#pragma once

#include "../../tensor/tensor.hpp"           // for Tensor
#include "../../tensorfuncs/adjoint_ops.hpp" // for tensorfuncs::adjoint
#include "../../tensorfuncs/kernels.hpp"     // for Kernels
#include "../../tensorfuncs/primal_ops.hpp"  // for tensorfuncs::primal
#include "../../tensorfuncs/strides.hpp"     // for flexible_binary
#include "dispatchers.hpp"                   // for Dispatchers
#include "exceptions.hpp"                    // for shape_error
#include <cstddef>                           // for size_t
#include <memory>                            // for std::make_unique

namespace kaad {

template <typename T> struct Computation_graph;
template <typename T> struct INode;
template <typename T, class Kernel> struct Node_binary;
template <typename T, class Kernel> struct Node_binary_flex;

namespace detail {

/**
 * @brief Contains a collection of binary functions for multiple versions
 * (sclalarRhs, scalarLhs, pointwise, flexible) of the operation and gradient of
 * a given binary Kernel.
 *
 * @tparam T Datatype the operations are performed on (e.g. float, double, ...).
 * @tparam Kernel Kernel the functions should be using.
 */
template <typename T, class Kernel> struct BinaryKernels {
    using Op = class Kernel::Op;
    using Grad = class Kernel::Grad;

    tensorfuncs::primal::binary::pointwise_fn<T, Op> scalarOpRhs =
        tensorfuncs::primal::binary::scalarRhs<T, Op>;
    tensorfuncs::primal::binary::pointwise_fn<T, Op> scalarOpLhs =
        tensorfuncs::primal::binary::scalarLhs<T, Op>;
    tensorfuncs::primal::binary::pointwise_fn<T, Op> pointOp =
        tensorfuncs::primal::binary::pointwise<T, Op>;
    tensorfuncs::primal::binary::flexible_fn<T, Op> flexOp =
        tensorfuncs::primal::binary::flexible<T, Op>;

    tensorfuncs::adjoint::binary::pointwise_fn<T, Grad> scalarGradRhs =
        tensorfuncs::adjoint::binary::scalarRhs<T, Grad>;
    tensorfuncs::adjoint::binary::pointwise_fn<T, Grad> scalarGradLhs =
        tensorfuncs::adjoint::binary::scalarLhs<T, Grad>;
    tensorfuncs::adjoint::binary::pointwise_fn<T, Grad> pointGrad =
        tensorfuncs::adjoint::binary::pointwise<T, Grad>;
    tensorfuncs::adjoint::binary::flexible_fn<T, Grad> flexGrad =
        tensorfuncs::adjoint::binary::flexible<T, Grad>;
};

/**
 * @internal
 * @brief Internal helper function not intended for direct user calls.
 *
 * Adds a generalized binary operation node to the computation graph `rec`.
 * Applies the binary operation specified by `kernels` to the input tensor nodes
 * `A_ptr` and `B_ptr`.
 *
 * @tparam T The data type of tensor elements.
 * @tparam Kernel The kernel providing forward operation and gradient.
 *
 * @param rec Reference to the computation graph.
 * @param A_ptr Pointer to the first input node.
 * @param B_ptr Pointer to the second input node.
 * @param kernels Binary operation and gradient kernels.
 * @param opName A string identifier for the operation (used for debugging or
 * logging).
 * @return Pointer to the newly created binary operation node.
 */
template <typename T, class Kernel>
INode<T> *binOperator(Computation_graph<T> &rec, INode<T> *A_ptr,
                      INode<T> *B_ptr, const BinaryKernels<T, Kernel> kernels,
                      const char *opName) {
    int recLen = rec.nodes.size();
    Tensor<T> &A = A_ptr->value;
    Tensor<T> &B = B_ptr->value;
    bool A_scalar = A.nDims() == 1 && A.shape[0] == 1;
    bool B_scalar = B.nDims() == 1 && B.shape[0] == 1;

    size_t newLen = std::max(A.nDims(), B.nDims());
    std::vector<int> newShape(newLen);

    if (B_scalar) {
        auto newNode = std::make_unique<Node_binary<T, Kernel>>(
            kernels.scalarOpRhs, kernels.scalarGradRhs, A_ptr, B_ptr, A.shape);
        auto raw_ptr = newNode.get();
        raw_ptr->end = raw_ptr->value.data() + raw_ptr->value.val.size();
        rec.nodes.push_back(std::move(newNode));
    } else if (A_scalar) {
        auto newNode = std::make_unique<Node_binary<T, Kernel>>(
            kernels.scalarOpLhs, kernels.scalarGradLhs, A_ptr, B_ptr, B.shape);
        auto raw_ptr = newNode.get();
        raw_ptr->end = raw_ptr->value.data() + raw_ptr->value.val.size();
        rec.nodes.push_back(std::move(newNode));
    } else if (A.nDims() == B.nDims() &&
               std::equal(A.shape.begin(), A.shape.end(), B.shape.data()) &&
               std::equal(A.stride.begin(), A.stride.end(), B.stride.data())) {

        auto newNode = std::make_unique<Node_binary<T, Kernel>>(
            kernels.pointOp, kernels.pointGrad, A_ptr, B_ptr, A.shape);
        auto raw_ptr = newNode.get();
        raw_ptr->end = raw_ptr->value.data() + raw_ptr->value.val.size();
        rec.nodes.push_back(std::move(newNode));
    } else if (combine_flexible(A.shape.data(), A.nDims(), B.shape.data(),
                                B.nDims(), newShape.data(), newLen)) {
        using Op = typename Kernel::Op;
        using Grad = typename Kernel::Grad;
        tensorfuncs::primal::binary::flexible_fn<T, Op> operation =
            kernels.flexOp;
        tensorfuncs::adjoint::binary::flexible_fn<T, Grad> gradient =
            kernels.flexGrad;
        if (newLen <= KAAD_MAX_NDIMS) {
            operation = detail::Dispatchers::get_flexOp<T, Op>()[newLen];
            gradient = detail::Dispatchers::get_flexGrad<T, Grad>()[newLen];
        }

        auto newNode = std::make_unique<Node_binary_flex<T, Kernel>>(
            A_ptr, B_ptr, newShape);
        Strides::flexible_binary<T>(*newNode.get());
        rec.nodes.push_back(std::move(newNode));
    } else {
        throw shape_error(recLen, opName,
                          "incompatible tensor shapes for binary operation",
                          {{"A.shape", A.shape}, {"B.shape", B.shape}});
    }
    return rec.nodes.back().get();
}

} // namespace detail

/**
 * @brief Adds a binary addition node (A + B) to the computation graph.
 *
 * Computes the element-wise sum of two input tensor nodes `A_ptr` and `B_ptr`.
 * Both tensors must have the same shape or be broadcast-compatible.
 *
 * @tparam T The data type of the tensor values.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A_ptr Pointer to the first input tensor node A.
 * @param B_ptr Pointer to the second input tensor node B.
 * @return A pointer to the new node representing the element-wise sum of A and
 * B.
 */
template <typename T>
INode<T> *add(Computation_graph<T> &rec, INode<T> *A_ptr, INode<T> *B_ptr) {
    static const detail::BinaryKernels<T, class Kernels::Add<T>> addK;
    return binOperator(rec, A_ptr, B_ptr, addK, "add");
}

/**
 * @brief Adds a binary subtratction node (A - B) to the computation graph.
 *
 * Computes the element-wise difference of two input tensor nodes `A_ptr` and
 * `B_ptr`. Both tensors must have the same shape or be broadcast-compatible.
 *
 * @tparam T The data type of the tensor values.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A_ptr Pointer to the first input tensor node A.
 * @param B_ptr Pointer to the second input tensor node B.
 * @return A pointer to the new node representing the element-wise difference of
 * A and B.
 */
template <typename T>
INode<T> *sub(Computation_graph<T> &rec, INode<T> *A_ptr, INode<T> *B_ptr) {
    static const detail::BinaryKernels<T, class Kernels::Sub<T>> subK;
    return binOperator(rec, A_ptr, B_ptr, subK, "sub");
}

/**
 * @brief Adds a binary multiplication node (A * B) to the computation
 * graph.
 *
 * Computes the element-wise product of two input tensor nodes `A_ptr` and
 * `B_ptr`. Both tensors must have the same shape or be broadcast-compatible.
 *
 * @tparam T The data type of the tensor values.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A_ptr Pointer to the first input tensor node A.
 * @param B_ptr Pointer to the second input tensor node B.
 * @return A pointer to the new node representing the element-wise product of A
 * and B.
 */
template <typename T>
INode<T> *mul(Computation_graph<T> &rec, INode<T> *A_ptr, INode<T> *B_ptr) {
    static const detail::BinaryKernels<T, class Kernels::Mul<T>> mulK;
    return binOperator(rec, A_ptr, B_ptr, mulK, "mul");
}

/**
 * @brief Adds a binary division node (A / B) to the computation graph.
 *
 * Computes the element-wise quotient of two input tensor nodes `A_ptr` and
 * `B_ptr`. Both tensors must have the same shape or be broadcast-compatible.
 *
 * @tparam T The data type of the tensor values.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A_ptr Pointer to the first input tensor node A.
 * @param B_ptr Pointer to the second input tensor node B.
 * @return A pointer to the new node representing the element-wise quotient of A
 * and B.
 */
template <typename T>
INode<T> *div(Computation_graph<T> &rec, INode<T> *A_ptr, INode<T> *B_ptr) {
    static const detail::BinaryKernels<T, class Kernels::Div<T>> divK;
    return binOperator(rec, A_ptr, B_ptr, divK, "div");
}

/**
 * @brief Adds a binary power node (A ^ B) to the computation graph.
 *
 * Computes the element-wise power of two input tensor nodes `A_ptr` and
 * `B_ptr`. Both tensors must have the same shape or be broadcast-compatible.
 *
 * @tparam T The data type of the tensor values.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A_ptr Pointer to the first input tensor node A.
 * @param B_ptr Pointer to the second input tensor node B.
 * @return A pointer to the new node representing the element-wise power of A
 * and B.
 */
template <typename T>
INode<T> *pow(Computation_graph<T> &rec, INode<T> *A_ptr, INode<T> *B_ptr) {
    static const detail::BinaryKernels<T, class Kernels::Pow<T>> powK;
    return binOperator(rec, A_ptr, B_ptr, powK, "pow");
}

/**
 * @brief Adds a binary minimum node (A __SYMBOL__ B) to the computation graph.
 *
 * Computes the element-wise minimum of two input tensor nodes `A_ptr` and
 * `B_ptr`. Both tensors must have the same shape or be broadcast-compatible.
 *
 * @tparam T The data type of the tensor values.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A_ptr Pointer to the first input tensor node A.
 * @param B_ptr Pointer to the second input tensor node B.
 * @return A pointer to the new node representing the element-wise minimum of A
 * and B.
 */
template <typename T>
INode<T> *min(Computation_graph<T> &rec, INode<T> *A_ptr, INode<T> *B_ptr) {
    static const detail::BinaryKernels<T, class Kernels::Min<T>> minK;
    return binOperator(rec, A_ptr, B_ptr, minK, "minimum");
}

/**
 * @brief Adds a binary maximum node (A __SYMBOL__ B) to the computation graph.
 *
 * Computes the element-wise maximum of two input tensor nodes `A_ptr` and
 * `B_ptr`. Both tensors must have the same shape or be broadcast-compatible.
 *
 * @tparam T The data type of the tensor values.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A_ptr Pointer to the first input tensor node A.
 * @param B_ptr Pointer to the second input tensor node B.
 * @return A pointer to the new node representing the element-wise maximum of A
 * and B.
 */
template <typename T>
INode<T> *max(Computation_graph<T> &rec, INode<T> *A_ptr, INode<T> *B_ptr) {
    static const detail::BinaryKernels<T, class Kernels::Max<T>> maxK;
    return binOperator(rec, A_ptr, B_ptr, maxK, "minimum");
}

} // namespace kaad
