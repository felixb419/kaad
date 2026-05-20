#include "../operations/broadcast.hpp"
#include "../operations/operation_concept.hpp"
#include "../operations/pointwise.hpp"
#include "kaad/operators/internal/kernels.hpp"
#include "kaad/operators/internal/safe_kernels.hpp"

#include <array>
#include <kaad/exceptions.hpp>
#include <kaad/graph/graph.hpp>
#include <kaad/graph/node_handle.hpp>
#include <kaad/operators/operators.hpp>
#include <kaad/scalar.hpp>
#include <memory>
#include <vector>

namespace kaad {

struct INode;
template <Operation operation> class OperatorNode;

template <operations::kernels::Binary Kernel>
Node binary_operator(Graph &graph, Node lhs, Node rhs, const char *opName) {

    INode *lhs_ptr = graph.get_node(lhs);
    INode *rhs_ptr = graph.get_node(rhs);

    try {

        // try pointwise operation
        graph.nodes.push_back(
            std::make_unique<
                OperatorNode<operations::Pointwise::Binary<Kernel>>>(
                std::array{lhs_ptr, rhs_ptr}));

    } catch (ShapeError &) {

        try {

            // fall back on broadcast operation
            graph.nodes.push_back(
                std::make_unique<OperatorNode<operations::Broadcast<Kernel>>>(
                    std::array{lhs_ptr, rhs_ptr}));

        } catch (BroadcastError &err) {

            throw BroadcastError(
                make_graph_errmsg(graph.nodes.size(), opName, err.what()));
        }
    };

    return graph.back_handle();
}

Node add(Graph &graph, Node lhs, Node rhs) {
    return binary_operator<operations::kernels::Add<Scalar>>(graph, lhs, rhs,
                                                             "add");
}

Node sub(Graph &graph, Node lhs, Node rhs) {
    return binary_operator<operations::kernels::Sub<Scalar>>(graph, lhs, rhs,
                                                             "sub");
}

Node mul(Graph &graph, Node lhs, Node rhs) {
    return binary_operator<operations::kernels::Mul<Scalar>>(graph, lhs, rhs,
                                                             "mul");
}

Node div(Graph &graph, Node lhs, Node rhs) {
    return binary_operator<operations::kernels::SafeDiv<Scalar>>(graph, lhs,
                                                                 rhs, "div");
}

Node pow(Graph &graph, Node lhs, Node rhs) {
    return binary_operator<operations::kernels::SafePow<Scalar>>(graph, lhs,
                                                                 rhs, "pow");
}

Node min(Graph &graph, Node lhs, Node rhs) {
    return binary_operator<operations::kernels::Min<Scalar>>(graph, lhs, rhs,
                                                             "minimum");
}

Node max(Graph &graph, Node lhs, Node rhs) {
    return binary_operator<operations::kernels::Max<Scalar>>(graph, lhs, rhs,
                                                             "minimum");
}

} // namespace kaad
