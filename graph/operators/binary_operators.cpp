#include "operators.hpp"

#include "../../exceptions.hpp"              // for shape_error
#include "../../scalar.hpp"                  // for Scalar
#include "../../tensor/tensor.hpp"           // for Tensor
#include "../../tensorfuncs/adjoint_ops.hpp" // for tensorfuncs::adjoint
#include "../../tensorfuncs/kernels.hpp"     // for Kernels
#include "../../tensorfuncs/primal_ops.hpp"  // for tensorfuncs::primal
#include "../computation_graph.hpp"          // for Computation_graph
#include "../node_handle.hpp"                // for Node_handle
#include "../nodes/binary.hpp"               // for Node_binary
#include "../nodes/binary_flex.hpp"          // for Node_binary_flex
#include "../nodes/inode.hpp"                // for INode
#include <cstddef>                           // for std::size_t
#include <memory>                            // for std::make_unique

namespace kaad {

/**
 * @brief Computes the resulting shape from broadcasting two tensors.
 * The broadcasting rules are:
 *   - dimensions must match
 *   - or one of them must be 1
 * @param shape1 Shape of first tensor.
 * @param rank1 Number of dimensions of first tensor.
 * @param shape2 Shape of second tensor.
 * @param rank2 Number of dimensions of second tensor.
 * @param newShape Output array to hold the result shape.
 * @param newLen Total number of dimensions in the result.
 * @return true if broadcasting is possible, false otherwise.
 */
static inline bool combine_flexible(const int *shape1, std::size_t rank1,
                                    const int *shape2, std::size_t rank2,
                                    int *newShape,
                                    std::size_t newLen) noexcept {
    int ind = newLen - 1;
    for (int i = 1; i <= newLen; i++, ind--) {
        int ind1 = rank1 - i;
        int ind2 = rank2 - i;
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
template <class Kernel> struct BinaryKernels {
    tensorfuncs::primal::binary::pointwise_fn<Kernel> scalarOpRhs =
        tensorfuncs::primal::binary::scalarRhs<Kernel>;
    tensorfuncs::primal::binary::pointwise_fn<Kernel> scalarOpLhs =
        tensorfuncs::primal::binary::scalarLhs<Kernel>;
    tensorfuncs::primal::binary::pointwise_fn<Kernel> pointOp =
        tensorfuncs::primal::binary::pointwise<Kernel>;
    tensorfuncs::primal::binary::flexible_fn<Kernel> flexOp =
        tensorfuncs::primal::binary::flexible<Kernel>;

    tensorfuncs::adjoint::binary::pointwise_fn<Kernel> scalarGradRhs =
        tensorfuncs::adjoint::binary::scalarRhs<Kernel>;
    tensorfuncs::adjoint::binary::pointwise_fn<Kernel> scalarGradLhs =
        tensorfuncs::adjoint::binary::scalarLhs<Kernel>;
    tensorfuncs::adjoint::binary::pointwise_fn<Kernel> pointGrad =
        tensorfuncs::adjoint::binary::pointwise<Kernel>;
    tensorfuncs::adjoint::binary::flexible_fn<Kernel> flexGrad =
        tensorfuncs::adjoint::binary::flexible<Kernel>;
};

/**
 * @internal
 * @brief Internal helper function not intended for direct user calls.
 *
 * Adds a generalized binary operation node to the computation graph `rec`.
 * Applies the binary operation specified by `kernels` to the input tensor nodes
 * `A` and `B`.
 *
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
template <class Kernel>
Node_handle binOperator(Computation_graph &rec, Node_handle A, Node_handle B,
                        const BinaryKernels<Kernel> kernels,
                        const char *opName) {
    int recLen = rec.nodes.size();

    INode *A_ptr = rec.get_node(A);
    INode *B_ptr = rec.get_node(B);
    Tensor &A_val = A_ptr->value;
    Tensor &B_val = B_ptr->value;

    bool A_scalar = A_val.rank() == 1 && A_val.shape()[0] == 1;
    bool B_scalar = B_val.rank() == 1 && B_val.shape()[0] == 1;

    std::size_t newLen = std::max(A_val.rank(), B_val.rank());
    std::vector<int> newShape(newLen);

    if (B_scalar) {

        rec.nodes.push_back(std::move(std::make_unique<Node_binary<Kernel>>(
            kernels.scalarOpRhs, kernels.scalarGradRhs, A_ptr, B_ptr,
            A_val.shape())));

    } else if (A_scalar) {

        rec.nodes.push_back(std::move(std::make_unique<Node_binary<Kernel>>(
            kernels.scalarOpLhs, kernels.scalarGradLhs, A_ptr, B_ptr,
            B_val.shape())));

    } else if (A_val.rank() == B_val.rank() &&
               std::equal(A_val.shape().begin(), A_val.shape().end(),
                          B_val.shape().begin()) &&
               std::equal(A_val.stride().begin(), A_val.stride().end(),
                          B_val.stride().begin())) {

        rec.nodes.push_back(std::move(std::make_unique<Node_binary<Kernel>>(
            kernels.pointOp, kernels.pointGrad, A_ptr, B_ptr, A_val.shape())));

    } else if (combine_flexible(A_val.shape().data(), A_val.rank(),
                                B_val.shape().data(), B_val.rank(),
                                newShape.data(), newLen)) {
        rec.nodes.push_back(
            std::move(std::make_unique<Node_binary_flex<Kernel>>(A_ptr, B_ptr,
                                                                 newShape)));
    } else {
        throw shape_error(make_graph_errmsg(
            "shape error", recLen, opName,
            "incompatible tensor shapes for binary operation",
            {{"A.shape", A_val.shape()}, {"B.shape", B_val.shape()}}));
    }
    return rec.back_handle();
}

Node_handle add(Computation_graph &rec, Node_handle A, Node_handle B) {
    static const BinaryKernels<Kernels::Add<Scalar>> addK;
    return binOperator(rec, A, B, addK, "add");
}

Node_handle sub(Computation_graph &rec, Node_handle A, Node_handle B) {
    static const BinaryKernels<Kernels::Sub<Scalar>> subK;
    return binOperator(rec, A, B, subK, "sub");
}

Node_handle mul(Computation_graph &rec, Node_handle A, Node_handle B) {
    static const BinaryKernels<Kernels::Mul<Scalar>> mulK;
    return binOperator(rec, A, B, mulK, "mul");
}

Node_handle div(Computation_graph &rec, Node_handle A, Node_handle B) {
    static const BinaryKernels<Kernels::Div<Scalar>> divK;
    return binOperator(rec, A, B, divK, "div");
}

Node_handle pow(Computation_graph &rec, Node_handle A, Node_handle B) {
    static const BinaryKernels<Kernels::Pow<Scalar>> powK;
    return binOperator(rec, A, B, powK, "pow");
}

Node_handle min(Computation_graph &rec, Node_handle A, Node_handle B) {
    static const BinaryKernels<Kernels::Min<Scalar>> minK;
    return binOperator(rec, A, B, minK, "minimum");
}

Node_handle max(Computation_graph &rec, Node_handle A, Node_handle B) {
    static const BinaryKernels<Kernels::Max<Scalar>> maxK;
    return binOperator(rec, A, B, maxK, "minimum");
}

} // namespace kaad
