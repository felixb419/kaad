#include "../../include/kaad/graph/nodes/dot.hpp"     // for Node_dot, dot
#include "../../include/kaad/exceptions.hpp"          // for make_graph_errmsg
#include "../../include/kaad/functions/adjoint.hpp"   // for scalarDot
#include "../../include/kaad/functions/primal.hpp"    // for scalarDot
#include "../../include/kaad/graph/graph.hpp"         // for Graph
#include "../../include/kaad/graph/node_handle.hpp"   // for Node
#include "../../include/kaad/graph/nodes/inode.hpp"   // for INode
#include "../../include/kaad/operators/operators.hpp" // for dot
#include "../../include/kaad/tensor/tensor.hpp"       // for Tensor
#include <algorithm>                                  // for equal
#include <memory>                                     // for unique_ptr, mak...
#include <utility>                                    // for move
#include <vector>                                     // for vector

namespace kaad {

Node dot(Graph &rec, Node lhs, Node rhs) {

    std::size_t recLen = rec.nodes.size();

    INode *lhs_ptr = rec.get_node(lhs);
    INode *rhs_ptr = rec.get_node(rhs);
    Tensor &lhs_val = lhs_ptr->value();
    Tensor &rhs_val = rhs_ptr->value();

    bool lhs_scalar = lhs_val.size() == 1;
    bool rhs_scalar = rhs_val.size() == 1;

    if (lhs_scalar || rhs_scalar) {

        if (rhs_scalar) {

            rec.nodes.push_back(std::make_unique<Node_dot>(lhs_ptr, rhs_ptr));

        } else if (lhs_scalar) {

            rec.nodes.push_back(std::make_unique<Node_dot>(rhs_ptr, lhs_ptr));
        }

        // override functions to ScalarDot
        static_cast<Node_dot *>(rec.nodes.back().get())->forward_op =
            functions::primal::binary::scalarDot;
        static_cast<Node_dot *>(rec.nodes.back().get())->backward_op =
            functions::adjoint::binary::scalarDot;

    } else if (lhs_val.rank() == 1 && rhs_val.rank() == 1 &&
               std::equal(lhs_val.shape().begin(), lhs_val.shape().end(),
                          rhs_val.shape().begin())) {

        rec.nodes.push_back(std::make_unique<Node_dot>(lhs_ptr, rhs_ptr));

    } else {
        throw shape_error(make_graph_errmsg(
            "shape error", recLen, "dot",
            "incompatible tensor shapes for dot product",
            {{"A.shape", lhs_val.shape()}, {"B.shape", rhs_val.shape()}}));
    }

    return rec.back_handle();
}

} // namespace kaad
