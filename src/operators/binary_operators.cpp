#include <kaad/operators/operators.hpp> // for add, div, max, min, mul

#include "../operations/safe_kernels.hpp"   // for SafeDiv, SafePow
#include <array>                            // for array
#include <kaad/exceptions.hpp>              // for BroadcastError, make_gra...
#include <kaad/graph/graph.hpp>             // for Graph, binary_operator
#include <kaad/graph/node_handle.hpp>       // for Node
#include <kaad/graph/operation_concept.hpp> // for Operation
#include <kaad/operations/flexible.hpp>     // for Flexible
#include <kaad/operations/kernels.hpp>      // for Add, Max, Min, Mul, Sub
#include <kaad/operations/pointwise.hpp>    // for Pointwise
#include <kaad/scalar.hpp>                  // for Scalar
#include <memory>                           // for make_unique, unique_ptr
#include <string>                           // for basic_string
#include <vector>                           // for vector

namespace kaad {

class INode;
template <Operation operation> class OperatorNode;

template <class Kernel>
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

            // fall back on flexible operation
            rec.nodes.push_back(
                std::make_unique<OperatorNode<operations::Flexible<Kernel>>>(
                    std::array{lhs_ptr, rhs_ptr}));

        } catch (BroadcastError &err) {

            throw BroadcastError(
                make_graph_errmsg(rec.nodes.size(), opName, err.what()));
        }
    };

    return rec.back_handle();
}

Node add(Graph &rec, Node lhs, Node rhs) {
    return binary_operator<Kernels::Add<Scalar>>(rec, lhs, rhs, "add");
}

Node sub(Graph &rec, Node lhs, Node rhs) {
    return binary_operator<Kernels::Sub<Scalar>>(rec, lhs, rhs, "sub");
}

Node mul(Graph &rec, Node lhs, Node rhs) {
    return binary_operator<Kernels::Mul<Scalar>>(rec, lhs, rhs, "mul");
}

Node div(Graph &rec, Node lhs, Node rhs) {
    return binary_operator<Kernels::SafeDiv<Scalar>>(rec, lhs, rhs, "div");
}

Node pow(Graph &rec, Node lhs, Node rhs) {
    return binary_operator<Kernels::SafePow<Scalar>>(rec, lhs, rhs, "pow");
}

Node min(Graph &rec, Node lhs, Node rhs) {
    return binary_operator<Kernels::Min<Scalar>>(rec, lhs, rhs, "minimum");
}

Node max(Graph &rec, Node lhs, Node rhs) {
    return binary_operator<Kernels::Max<Scalar>>(rec, lhs, rhs, "minimum");
}

} // namespace kaad
