#include "../../operations/operation_concept.hpp"
#include "../../operations/pointwise.hpp"
#include "kaad/graph/operators/internal/kernels.hpp"
#include "kaad/graph/operators/internal/safe_kernels.hpp"

#include <array>
#include <kaad/graph/graph.hpp>
#include <kaad/graph/node_handle.hpp>
#include <kaad/scalar.hpp>

namespace kaad {

template <Operation operation> class OperatorNode;

template <operations::kernels::Unary Kernel>
Node Graph::unary_operator(Node input) {

    this->nodes.push_back(
        new OperatorNode<operations::Pointwise::Unary<Kernel>>(
            std::array{this->get_node(input)}));

    return Node(this->nodes.size() - 1, this);
}

Node Graph::negative(Node input) {
    return this->unary_operator<operations::kernels::Neg<Scalar>>(input);
}

Node Graph::square(Node input) {
    return this->unary_operator<operations::kernels::Square<Scalar>>(input);
}

Node Graph::sqrt(Node input) {
    return this->unary_operator<operations::kernels::SafeSqrt<Scalar>>(input);
}

Node Graph::log(Node input) {
    return this->unary_operator<operations::kernels::SafeLog<Scalar>>(input);
}

Node Graph::exp(Node input) {
    return this->unary_operator<operations::kernels::SafeExp<Scalar>>(input);
}

Node Graph::abs(Node input) {
    return this->unary_operator<operations::kernels::Abs<Scalar>>(input);
}

} // namespace kaad
