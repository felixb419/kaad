#include <kaad/operators/operators.hpp>

#include <kaad/functions/adjoint.hpp>      // for pointwise
#include <kaad/functions/kernels.hpp>      // for Abs, Neg
#include <kaad/functions/primal.hpp>       // for pointwise
#include <kaad/functions/safe_kernels.hpp> // for safe_Exp
#include <kaad/graph/graph.hpp>            // for Graph, unOp...
#include <kaad/graph/node_handle.hpp>      // for Node
#include <kaad/graph/nodes/inode.hpp>      // for INode
#include <kaad/graph/nodes/unary.hpp>      // for Node_unary
#include <kaad/scalar.hpp>                 // for Scalar
#include <kaad/tensor/tensor.hpp>          // for Tensor
#include <memory>                          // for make_unique
#include <utility>                         // for move

// IWYU pragma: no_forward_declare kaad::Node_unary

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
 * `input`.
 *
 * @tparam Kernel The kernel providing forward operation and gradient.
 * @param rec Reference to the computation graph.
 * @param input Handle of the input node.
 * @param kernels Unary operation and gradient kernels.
 * @return Handle of the newly created unary operation node.
 */
template <class Kernel> Node unOperator(Graph &rec, Node input) {

    static const UnaryKernels<Kernel> kernels;

    INode *input_ptr = rec.get_node(input);
    Tensor &input_val = input_ptr->value();

    rec.nodes.push_back(std::move(std::make_unique<Node_unary<Kernel>>(
        kernels.op, kernels.grad, input_ptr, input_val.shape())));

    return rec.back_handle();
}

Node negative(Graph &rec, Node input) {
    return unOperator<Kernels::Neg<Scalar>>(rec, input);
}

Node square(Graph &rec, Node input) {
    return unOperator<Kernels::Square<Scalar>>(rec, input);
}

Node sqrt(Graph &rec, Node input) {
    return unOperator<Kernels::safe_Sqrt<Scalar>>(rec, input);
}

Node log(Graph &rec, Node input) {
    return unOperator<Kernels::safe_Log<Scalar>>(rec, input);
}

Node exp(Graph &rec, Node input) {
    return unOperator<Kernels::safe_Exp<Scalar>>(rec, input);
}

Node abs(Graph &rec, Node input) {
    return unOperator<Kernels::Abs<Scalar>>(rec, input);
}

} // namespace kaad
