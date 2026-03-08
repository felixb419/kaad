#include "../../include/kaad/exceptions.hpp"        // for make_graph_errmsg
#include "../../include/kaad/functions/adjoint.hpp" // for pointwise_fn, flexible
#include "../../include/kaad/functions/kernels.hpp" // for Add, Max, Min, Mul
#include "../../include/kaad/functions/primal.hpp" // for pointwise_fn, flexible
#include "../../include/kaad/functions/safe_kernels.hpp" // for safe_Div, safe_Pow
#include "../../include/kaad/graph/graph.hpp"            // for Graph
#include "../../include/kaad/graph/node_handle.hpp"      // for Node
#include "../../include/kaad/graph/nodes/binary.hpp"     // for Node_binary
#include "../../include/kaad/graph/nodes/binary_flex.hpp" // for Node_binary_flex
#include "../../include/kaad/graph/nodes/inode.hpp"       // for INode
#include "../../include/kaad/operators/operators.hpp" // for add, div, max, min
#include "../../include/kaad/scalar.hpp"              // for Scalar
#include "../../include/kaad/tensor/tensor.hpp"       // for Tensor
#include <algorithm>                                  // for equal, max
#include <cstddef>                                    // for size_t
#include <memory>                                     // for make_unique
#include <utility>                                    // for move
#include <vector>                                     // for vector

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
    for (std::size_t i = 1; i <= newLen; i++, ind--) {
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
    functions::primal::binary::pointwise_fn<Kernel> scalarOpRhs =
        functions::primal::binary::scalarRhs<Kernel>;
    functions::primal::binary::pointwise_fn<Kernel> scalarOpLhs =
        functions::primal::binary::scalarLhs<Kernel>;
    functions::primal::binary::pointwise_fn<Kernel> pointOp =
        functions::primal::binary::pointwise<Kernel>;
    functions::primal::binary::flexible_fn<Kernel> flexOp =
        functions::primal::binary::flexible<Kernel>;

    functions::adjoint::binary::pointwise_fn<Kernel> scalarGradRhs =
        functions::adjoint::binary::scalarRhs<Kernel>;
    functions::adjoint::binary::pointwise_fn<Kernel> scalarGradLhs =
        functions::adjoint::binary::scalarLhs<Kernel>;
    functions::adjoint::binary::pointwise_fn<Kernel> pointGrad =
        functions::adjoint::binary::pointwise<Kernel>;
    functions::adjoint::binary::flexible_fn<Kernel> flexGrad =
        functions::adjoint::binary::flexible<Kernel>;
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
Node binOperator(Graph &rec, Node A, Node B, const char *opName) {

    static const BinaryKernels<Kernel> kernels;
    int recLen = rec.nodes.size();

    INode *A_ptr = rec.get_node(A);
    INode *B_ptr = rec.get_node(B);
    Tensor &A_val = A_ptr->value();
    Tensor &B_val = B_ptr->value();

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

Node add(Graph &rec, Node A, Node B) {
    return binOperator<Kernels::Add<Scalar>>(rec, A, B, "add");
}

Node sub(Graph &rec, Node A, Node B) {
    return binOperator<Kernels::Sub<Scalar>>(rec, A, B, "sub");
}

Node mul(Graph &rec, Node A, Node B) {
    return binOperator<Kernels::Mul<Scalar>>(rec, A, B, "mul");
}

Node div(Graph &rec, Node A, Node B) {
    return binOperator<Kernels::safe_Div<Scalar>>(rec, A, B, "div");
}

Node pow(Graph &rec, Node A, Node B) {
    return binOperator<Kernels::safe_Pow<Scalar>>(rec, A, B, "pow");
}

Node min(Graph &rec, Node A, Node B) {
    return binOperator<Kernels::Min<Scalar>>(rec, A, B, "minimum");
}

Node max(Graph &rec, Node A, Node B) {
    return binOperator<Kernels::Max<Scalar>>(rec, A, B, "minimum");
}

} // namespace kaad
