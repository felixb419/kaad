#include <kaad/operators/operators.hpp>

#include "../operations/operation_concept.hpp"
#include "../operations/pointwise.hpp"
#include "kaad/operators/internal/kernels.hpp"
#include "kaad/operators/internal/safe_kernels.hpp"

#include <array>
#include <kaad/graph/graph.hpp>
#include <kaad/graph/node_handle.hpp>
#include <kaad/scalar.hpp>
#include <memory>

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
