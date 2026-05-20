#include "../../operations/broadcast.hpp"
#include "../../operations/operation_concept.hpp"
#include "../../operations/pointwise.hpp"
#include "kaad/graph/operators/internal/kernels.hpp"
#include "kaad/graph/operators/internal/safe_kernels.hpp"

#include <array>
#include <kaad/exceptions.hpp>
#include <kaad/graph/graph.hpp>
#include <kaad/graph/node_handle.hpp>
#include <kaad/scalar.hpp>
#include <vector>

namespace kaad {

struct INode;
template <Operation operation> class OperatorNode;

template <operations::kernels::Binary Kernel>
Node Graph::binary_operator(Node lhs, Node rhs) {

    INode *lhs_ptr = this->get_node(lhs);
    INode *rhs_ptr = this->get_node(rhs);

    try {

        // try pointwise operation
        this->nodes.push_back(
            new OperatorNode<operations::Pointwise::Binary<Kernel>>(
                std::array{lhs_ptr, rhs_ptr}));

    } catch (ShapeError &) {

        try {

            // fall back on broadcast operation
            this->nodes.push_back(
                new OperatorNode<operations::Broadcast<Kernel>>(
                    std::array{lhs_ptr, rhs_ptr}));

        } catch (BroadcastError &err) {

            throw BroadcastError(make_graph_errmsg(
                this->nodes.size(), Kernel::OPERATION_NAME, err.what()));
        }
    };

    return Node(this->nodes.size() - 1, this);
}

Node Graph::add(Node lhs, Node rhs) {
    return this->binary_operator<operations::kernels::Add<Scalar>>(lhs, rhs);
}

Node Graph::sub(Node lhs, Node rhs) {
    return this->binary_operator<operations::kernels::Sub<Scalar>>(lhs, rhs);
}

Node Graph::mul(Node lhs, Node rhs) {
    return this->binary_operator<operations::kernels::Mul<Scalar>>(lhs, rhs);
}

Node Graph::div(Node lhs, Node rhs) {
    return this->binary_operator<operations::kernels::SafeDiv<Scalar>>(lhs,
                                                                       rhs);
}

Node Graph::pow(Node lhs, Node rhs) {
    return this->binary_operator<operations::kernels::SafePow<Scalar>>(lhs,
                                                                       rhs);
}

Node Graph::min(Node lhs, Node rhs) {
    return this->binary_operator<operations::kernels::Min<Scalar>>(lhs, rhs);
}

Node Graph::max(Node lhs, Node rhs) {
    return this->binary_operator<operations::kernels::Max<Scalar>>(lhs, rhs);
}

} // namespace kaad
