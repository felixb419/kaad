#include <kaad/operators/operators.hpp>

#include "../operations/broadcast.hpp"
#include "../operations/operation_concept.hpp"
#include "../operations/pointwise.hpp"
#include "kaad/operators/internal/kernels.hpp"
#include "kaad/operators/internal/safe_kernels.hpp"

#include <array>
#include <kaad/exceptions.hpp>
#include <kaad/graph/graph.hpp>
#include <kaad/graph/node_handle.hpp>
#include <kaad/scalar.hpp>
#include <memory>
#include <vector>

namespace kaad {

struct INode;
template <Operation operation> class OperatorNode;

template <operations::kernels::Binary Kernel>
Node binary_operator(Graph &rec, Node lhs, Node rhs, const char *opName) {

    INode *lhs_ptr = rec.get_node(lhs);
    INode *rhs_ptr = rec.get_node(rhs);

    try {

        // try pointwise operation
        rec.nodes.push_back(
            std::make_unique<
                OperatorNode<operations::Pointwise::Binary<Kernel>>>(
                std::array{lhs_ptr, rhs_ptr}));

    } catch (ShapeError &) {

        try {

            // fall back on broadcast operation
            rec.nodes.push_back(
                std::make_unique<OperatorNode<operations::Broadcast<Kernel>>>(
                    std::array{lhs_ptr, rhs_ptr}));

        } catch (BroadcastError &err) {

            throw BroadcastError(
                make_graph_errmsg(rec.nodes.size(), opName, err.what()));
        }
    };

    return rec.back_handle();
}

Node add(Graph &rec, Node lhs, Node rhs) {
    return binary_operator<operations::kernels::Add<Scalar>>(rec, lhs, rhs,
                                                             "add");
}

Node sub(Graph &rec, Node lhs, Node rhs) {
    return binary_operator<operations::kernels::Sub<Scalar>>(rec, lhs, rhs,
                                                             "sub");
}

Node mul(Graph &rec, Node lhs, Node rhs) {
    return binary_operator<operations::kernels::Mul<Scalar>>(rec, lhs, rhs,
                                                             "mul");
}

Node div(Graph &rec, Node lhs, Node rhs) {
    return binary_operator<operations::kernels::SafeDiv<Scalar>>(rec, lhs, rhs,
                                                                 "div");
}

Node pow(Graph &rec, Node lhs, Node rhs) {
    return binary_operator<operations::kernels::SafePow<Scalar>>(rec, lhs, rhs,
                                                                 "pow");
}

Node min(Graph &rec, Node lhs, Node rhs) {
    return binary_operator<operations::kernels::Min<Scalar>>(rec, lhs, rhs,
                                                             "minimum");
}

Node max(Graph &rec, Node lhs, Node rhs) {
    return binary_operator<operations::kernels::Max<Scalar>>(rec, lhs, rhs,
                                                             "minimum");
}

} // namespace kaad
