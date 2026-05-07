#include <kaad/operators/operators.hpp> // for abs, exp, log, negative, sqrt

#include "../operations/safe_kernels.hpp"   // for SafeExp, SafeLog, SafeSqrt
#include "kaad/graph/operation_concept.hpp" // for Operation
#include <array>                            // for array
#include <kaad/graph/graph.hpp>             // for Graph, unary_operator
#include <kaad/graph/node_handle.hpp>       // for Node
#include <kaad/operations/kernels.hpp>      // for Abs, Neg, Square
#include <kaad/operations/pointwise.hpp>    // for Pointwise
#include <kaad/scalar.hpp>                  // for Scalar
#include <memory>                           // for make_unique

namespace kaad {

template <Operation operation> class OperatorNode;

template <operations::kernels::Unary Kernel>
Node unary_operator(Graph &rec, Node input) {

    rec.nodes.push_back(
        std::make_unique<OperatorNode<operations::Pointwise::Unary<Kernel>>>(
            std::array{rec.get_node(input)}));

    return rec.back_handle();
}

Node negative(Graph &rec, Node input) {
    return unary_operator<operations::kernels::Neg<Scalar>>(rec, input);
}

Node square(Graph &rec, Node input) {
    return unary_operator<operations::kernels::Square<Scalar>>(rec, input);
}

Node sqrt(Graph &rec, Node input) {
    return unary_operator<operations::kernels::SafeSqrt<Scalar>>(rec, input);
}

Node log(Graph &rec, Node input) {
    return unary_operator<operations::kernels::SafeLog<Scalar>>(rec, input);
}

Node exp(Graph &rec, Node input) {
    return unary_operator<operations::kernels::SafeExp<Scalar>>(rec, input);
}

Node abs(Graph &rec, Node input) {
    return unary_operator<operations::kernels::Abs<Scalar>>(rec, input);
}

} // namespace kaad
