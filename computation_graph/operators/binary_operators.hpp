#pragma once

#include "../../tensor/tensor.hpp"           // for Tensor
#include "../../tensorfuncs/adjoint_ops.hpp" // for tensorfuncs::adjoint
#include "../../tensorfuncs/kernels.hpp"     // for Kernels
#include "../../tensorfuncs/primal_ops.hpp"  // for tensorfuncs::primal
#include "../computation_graph.hpp"          // for Computation_graph
#include "../node_handle.hpp"                // for Node_handle
#include "../nodes/binary.hpp"               // for Node_binary
#include "../nodes/binary_flex.hpp"          // for Node_binary_flex
#include "../nodes/inode.hpp"                // for INode
#include "exceptions.hpp"                    // for shape_error
#include <cstddef>                           // for size_t
#include <memory>                            // for std::make_unique

namespace kaad {

namespace detail {

/**
 * @brief Computes the resulting shape from broadcasting two tensors.
 * The broadcasting rules are:
 *   - dimensions must match
 *   - or one of them must be 1
 * @param shape1 Shape of first tensor.
 * @param nDims1 Number of dimensions of first tensor.
 * @param shape2 Shape of second tensor.
 * @param nDims2 Number of dimensions of second tensor.
 * @param newShape Output array to hold the result shape.
 * @param newLen Total number of dimensions in the result.
 * @return true if broadcasting is possible, false otherwise.
 */
static inline bool combine_flexible(const int *shape1, size_t nDims1,
                                    const int *shape2, size_t nDims2,
                                    int *newShape, size_t newLen) {
    int ind = newLen - 1;
    for (int i = 1; i <= newLen; i++, ind--) {
        int ind1 = nDims1 - i;
        int ind2 = nDims2 - i;
        if (ind1 >= 0 && ind2 >= 0) {
            if (shape1[ind1] != shape2[ind2] && shape1[ind1] != 1 &&
                shape2[ind2] != 1) {
                return false;
            }
            newShape[ind] = std::max(shape1[ind1], shape2[ind2]);
        } else {
            newShape[ind] = ind1 >= 0 ? shape1[ind1] : shape2[ind2];
        }
    }
    return true;
}

/**
 * @brief Contains a collection of binary functions for multiple versions
 * (sclalarRhs, scalarLhs, pointwise, flexible) of the operation and gradient of
 * a given binary Kernel.
 *
 * @tparam T Datatype the operations are performed on (e.g. float, double, ...).
 * @tparam Kernel Kernel the functions should be using.
 */
template <typename T, class Kernel> struct BinaryKernels {
    tensorfuncs::primal::binary::pointwise_fn<T, Kernel> scalarOpRhs =
        tensorfuncs::primal::binary::scalarRhs<T, Kernel>;
    tensorfuncs::primal::binary::pointwise_fn<T, Kernel> scalarOpLhs =
        tensorfuncs::primal::binary::scalarLhs<T, Kernel>;
    tensorfuncs::primal::binary::pointwise_fn<T, Kernel> pointOp =
        tensorfuncs::primal::binary::pointwise<T, Kernel>;
    tensorfuncs::primal::binary::flexible_fn<T, Kernel> flexOp =
        tensorfuncs::primal::binary::flexible<T, Kernel>;

    tensorfuncs::adjoint::binary::pointwise_fn<T, Kernel> scalarGradRhs =
        tensorfuncs::adjoint::binary::scalarRhs<T, Kernel>;
    tensorfuncs::adjoint::binary::pointwise_fn<T, Kernel> scalarGradLhs =
        tensorfuncs::adjoint::binary::scalarLhs<T, Kernel>;
    tensorfuncs::adjoint::binary::pointwise_fn<T, Kernel> pointGrad =
        tensorfuncs::adjoint::binary::pointwise<T, Kernel>;
    tensorfuncs::adjoint::binary::flexible_fn<T, Kernel> flexGrad =
        tensorfuncs::adjoint::binary::flexible<T, Kernel>;
};

/**
 * @internal
 * @brief Internal helper function not intended for direct user calls.
 *
 * Adds a generalized binary operation node to the computation graph `rec`.
 * Applies the binary operation specified by `kernels` to the input tensor nodes
 * `A` and `B`.
 *
 * @tparam T The data type of tensor elements.
 * @tparam Kernel The kernel providing forward operation and gradient.
 *
 * @param rec Reference to the computation graph.
 * @param A Handle of the first input node.
 * @param B Handle of the second input node.
 * @param kernels Binary operation and gradient kernels.
 * @param opName A string identifier for the operation (used for debugging or
 * logging).
 * @return Handle of the newly created binary operation node.
 */
template <typename T, class Kernel>
Node_handle binOperator(Computation_graph &rec, Node_handle A, Node_handle B,
                        const BinaryKernels<T, Kernel> kernels,
                        const char *opName) {
    int recLen = rec.nodes.size();

    INode *A_ptr = rec.get_node(A);
    INode *B_ptr = rec.get_node(B);
    Tensor &A_val = A_ptr->value;
    Tensor &B_val = B_ptr->value;

    bool A_scalar = A_val.nDims() == 1 && A_val.shape()[0] == 1;
    bool B_scalar = B_val.nDims() == 1 && B_val.shape()[0] == 1;

    size_t newLen = std::max(A_val.nDims(), B_val.nDims());
    std::vector<int> newShape(newLen);

    if (B_scalar) {

        rec.nodes.push_back(std::move(std::make_unique<Node_binary<Kernel>>(
            kernels.scalarOpRhs, kernels.scalarGradRhs, A_ptr, B_ptr,
            A_val.shape())));

    } else if (A_scalar) {

        rec.nodes.push_back(std::move(std::make_unique<Node_binary<Kernel>>(
            kernels.scalarOpLhs, kernels.scalarGradLhs, A_ptr, B_ptr,
            B_val.shape())));

    } else if (A_val.nDims() == B_val.nDims() &&
               std::equal(A_val.shape_begin(), A_val.shape_end(),
                          B_val.shape_begin()) &&
               std::equal(A_val.stride_begin(), A_val.stride_end(),
                          B_val.stride_begin())) {

        rec.nodes.push_back(std::move(std::make_unique<Node_binary<Kernel>>(
            kernels.pointOp, kernels.pointGrad, A_ptr, B_ptr, A_val.shape())));

    } else if (combine_flexible(A_val.shape_begin(), A_val.nDims(),
                                B_val.shape_begin(), B_val.nDims(),
                                newShape.data(), newLen)) {
        rec.nodes.push_back(
            std::move(std::make_unique<Node_binary_flex<Kernel>>(A_ptr, B_ptr,
                                                                 newShape)));
    } else {
        throw shape_error(
            recLen, opName, "incompatible tensor shapes for binary operation",
            {{"A.shape", A_val.shape()}, {"B.shape", B_val.shape()}});
    }
    return rec.back_handle();
}

} // namespace detail

/**
 * @brief Adds a binary addition node (A + B) to the computation graph.
 *
 * Computes the element-wise sum of two input tensor nodes `A` and `B`.
 * Both tensors must have the same shape or be broadcast-compatible.
 *
 * @tparam T The data type of the tensor values.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A Handle of the first input tensor node A.
 * @param B Handle of the second input tensor node B.
 * @return A handle of the new node representing the element-wise sum of A and
 * B.
 */
template <typename T>
Node_handle add(Computation_graph &rec, Node_handle A, Node_handle B) {
    static const detail::BinaryKernels<T, class Kernels::Add<T>> addK;
    return binOperator(rec, A, B, addK, "add");
}

/**
 * @brief Adds a binary subtratction node (A - B) to the computation graph.
 *
 * Computes the element-wise difference of two input tensor nodes `A` and
 * `B`. Both tensors must have the same shape or be broadcast-compatible.
 *
 * @tparam T The data type of the tensor values.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A Handle of the first input tensor node A.
 * @param B Handle of the second input tensor node B.
 * @return A handle of the new node representing the element-wise difference of
 * A and B.
 */
template <typename T>
Node_handle sub(Computation_graph &rec, Node_handle A, Node_handle B) {
    static const detail::BinaryKernels<T, class Kernels::Sub<T>> subK;
    return binOperator(rec, A, B, subK, "sub");
}

/**
 * @brief Adds a binary multiplication node (A * B) to the computation
 * graph.
 *
 * Computes the element-wise product of two input tensor nodes `A` and
 * `B`. Both tensors must have the same shape or be broadcast-compatible.
 *
 * @tparam T The data type of the tensor values.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A Handle of the first input tensor node A.
 * @param B Handle of the second input tensor node B.
 * @return A handle of the new node representing the element-wise product of A
 * and B.
 */
template <typename T>
Node_handle mul(Computation_graph &rec, Node_handle A, Node_handle B) {
    static const detail::BinaryKernels<T, class Kernels::Mul<T>> mulK;
    return binOperator(rec, A, B, mulK, "mul");
}

/**
 * @brief Adds a binary division node (A / B) to the computation graph.
 *
 * Computes the element-wise quotient of two input tensor nodes `A` and
 * `B`. Both tensors must have the same shape or be broadcast-compatible.
 *
 * @tparam T The data type of the tensor values.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A Handle of the first input tensor node A.
 * @param B Handle of the second input tensor node B.
 * @return A handle of the new node representing the element-wise quotient of A
 * and B.
 */
template <typename T>
Node_handle div(Computation_graph &rec, Node_handle A, Node_handle B) {
    static const detail::BinaryKernels<T, class Kernels::Div<T>> divK;
    return binOperator(rec, A, B, divK, "div");
}

/**
 * @brief Adds a binary power node (A ^ B) to the computation graph.
 *
 * Computes the element-wise power of two input tensor nodes `A` and
 * `B`. Both tensors must have the same shape or be broadcast-compatible.
 *
 * @tparam T The data type of the tensor values.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A Handle of the first input tensor node A.
 * @param B Handle of the second input tensor node B.
 * @return A handle of the new node representing the element-wise power of A
 * and B.
 */
template <typename T>
Node_handle pow(Computation_graph &rec, Node_handle A, Node_handle B) {
    static const detail::BinaryKernels<T, class Kernels::Pow<T>> powK;
    return binOperator(rec, A, B, powK, "pow");
}

/**
 * @brief Adds a binary minimum node (A __SYMBOL__ B) to the computation graph.
 *
 * Computes the element-wise minimum of two input tensor nodes `A` and
 * `B`. Both tensors must have the same shape or be broadcast-compatible.
 *
 * @tparam T The data type of the tensor values.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A Handle of the first input tensor node A.
 * @param B Handle of the second input tensor node B.
 * @return A handle of the new node representing the element-wise minimum of A
 * and B.
 */
template <typename T>
Node_handle min(Computation_graph &rec, Node_handle A, Node_handle B) {
    static const detail::BinaryKernels<T, class Kernels::Min<T>> minK;
    return binOperator(rec, A, B, minK, "minimum");
}

/**
 * @brief Adds a binary maximum node (A __SYMBOL__ B) to the computation graph.
 *
 * Computes the element-wise maximum of two input tensor nodes `A` and
 * `B`. Both tensors must have the same shape or be broadcast-compatible.
 *
 * @tparam T The data type of the tensor values.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A Handle of the first input tensor node A.
 * @param B Handle of the second input tensor node B.
 * @return A handle of the new node representing the element-wise maximum of A
 * and B.
 */
template <typename T>
Node_handle max(Computation_graph &rec, Node_handle A, Node_handle B) {
    static const detail::BinaryKernels<T, class Kernels::Max<T>> maxK;
    return binOperator(rec, A, B, maxK, "minimum");
}

} // namespace kaad
