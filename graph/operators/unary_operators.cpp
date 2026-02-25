#include "../../functions/adjoint.hpp"      // for pointwise, poin...
#include "../../functions/kernels.hpp"      // for Abs, Neg, Square
#include "../../functions/primal.hpp"       // for pointwise, poin...
#include "../../functions/safe_kernels.hpp" // for safe_Exp, safe_Log
#include "../../graph/nodes/inode.hpp"      // for INode
#include "../../scalar.hpp"                 // for Scalar
#include "../../tensor/tensor.hpp"          // for Tensor
#include "../computation_graph.hpp"         // for Computation_graph
#include "../node_handle.hpp"               // for Node_handle
#include "../nodes/unary.hpp"               // for Node_unary
#include "operators.hpp"                    // for abs, exp, log
#include <memory>                           // for make_unique
#include <utility>                          // for move

namespace kaad {

/**
 * @brief Contains a collection of unary functions for pointwise version of the
 * operation and gradient of a given unary Kernel.
 *
 * @tparam T Datatype the operations are performed on (e.g. float, double, ...).
 * @tparam Kernel Kernel the functions should be using.
 */
template <class Kernel> struct UnaryKernels {
    functions::primal::unary::pointwise_fn<Kernel> op =
        functions::primal::unary::pointwise<Kernel>;
    functions::adjoint::unary::pointwise_fn<Kernel> grad =
        functions::adjoint::unary::pointwise<Kernel>;
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
