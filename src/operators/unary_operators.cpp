#include "../operations/operation_concept.hpp"
#include "../operations/pointwise.hpp"
#include "kaad/operators/internal/kernels.hpp"
#include "kaad/operators/internal/safe_kernels.hpp"

#include <array>
#include <kaad/graph/graph.hpp>
#include <kaad/graph/node_handle.hpp>
#include <kaad/operators/operators.hpp>
#include <kaad/scalar.hpp>
#include <memory>

namespace kaad {

template <Operation operation> class OperatorNode;

template <operations::kernels::Unary Kernel>
Node unary_operator(Graph &graph, Node input) {

    graph.nodes.push_back(
        std::make_unique<OperatorNode<operations::Pointwise::Unary<Kernel>>>(
            std::array{graph.get_node(input)}));

    return graph.back_handle();
}

Node negative(Graph &graph, Node input) {
    return unary_operator<operations::kernels::Neg<Scalar>>(graph, input);
}

Node square(Graph &graph, Node input) {
    return unary_operator<operations::kernels::Square<Scalar>>(graph, input);
}

Node sqrt(Graph &graph, Node input) {
    return unary_operator<operations::kernels::SafeSqrt<Scalar>>(graph, input);
}

Node log(Graph &graph, Node input) {
    return unary_operator<operations::kernels::SafeLog<Scalar>>(graph, input);
}

Node exp(Graph &graph, Node input) {
    return unary_operator<operations::kernels::SafeExp<Scalar>>(graph, input);
}

Node abs(Graph &graph, Node input) {
    return unary_operator<operations::kernels::Abs<Scalar>>(graph, input);
}

} // namespace kaad
