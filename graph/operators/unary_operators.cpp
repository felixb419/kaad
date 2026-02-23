#include "operators.hpp"

#include "../../tensor/tensor.hpp"            // for Tensor
#include "../../tensorfuncs/adjoint_ops.hpp"  // for tensorfuncs::adjoint
#include "../../tensorfuncs/kernels.hpp"      // for Kernels
#include "../../tensorfuncs/primal_ops.hpp"   // for tensorfuncs::primal
#include "../../tensorfuncs/safe_kernels.hpp" // for Kernels
#include "../computation_graph.hpp"           // for Computation_graph
#include "../node_handle.hpp"                 // for Node_handle
#include "../nodes/unary.hpp"                 // for Node_unarysewfease
#include <algorithm>                          // for std::copy, std::fill
#include <memory>                             // for std::make_unique

namespace kaad {

/**
 * @brief Contains a collection of unary functions for pointwise version of the
 * operation and gradient of a given unary Kernel.
 *
 * @tparam T Datatype the operations are performed on (e.g. float, double, ...).
 * @tparam Kernel Kernel the functions should be using.
 */
template <class Kernel> struct UnaryKernels {
    tensorfuncs::primal::unary::pointwise_fn<Kernel> op =
        tensorfuncs::primal::unary::pointwise<Kernel>;
    tensorfuncs::adjoint::unary::pointwise_fn<Kernel> grad =
        tensorfuncs::adjoint::unary::pointwise<Kernel>;
};

/**
 * @internal
 * @brief Internal helper function not intended for direct user calls.
 *
 * Adds a generalized unary operation node to the computation graph `rec`.
 * Applies the unary operation specified by `kernels` to the input tensor node
 * `A`.
 *
 * @tparam Kernel The kernel providing forward operation and gradient.
 * @param rec Reference to the computation graph.
 * @param A Handle of the input node.
 * @param kernels Unary operation and gradient kernels.
 * @return Handle of the newly created unary operation node.
 */
template <class Kernel>
Node_handle unOperator(Computation_graph &rec, Node_handle A) {

    static const UnaryKernels<Kernel> kernels;

    INode *A_ptr = rec.get_node(A);
    Tensor &A_val = A_ptr->value();

    rec.nodes.push_back(std::move(std::make_unique<Node_unary<Kernel>>(
        kernels.op, kernels.grad, A_ptr, A_val.shape())));

    return rec.back_handle();
}

Node_handle negative(Computation_graph &rec, Node_handle A) {
    return unOperator<Kernels::Neg<Scalar>>(rec, A);
}

Node_handle square(Computation_graph &rec, Node_handle A) {
    return unOperator<Kernels::Square<Scalar>>(rec, A);
}

Node_handle sqrt(Computation_graph &rec, Node_handle A) {
    return unOperator<Kernels::safe_Sqrt<Scalar>>(rec, A);
}

Node_handle log(Computation_graph &rec, Node_handle A) {
    return unOperator<Kernels::safe_Log<Scalar>>(rec, A);
}

Node_handle exp(Computation_graph &rec, Node_handle A) {
    return unOperator<Kernels::safe_Exp<Scalar>>(rec, A);
}

Node_handle abs(Computation_graph &rec, Node_handle A) {
    return unOperator<Kernels::Abs<Scalar>>(rec, A);
}

} // namespace kaad
