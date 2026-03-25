#include <kaad/operators/operators.hpp> // for dot

#include "../exceptions.hpp"          // for ShapeError, make_graph_errmsg
#include "../graph/nodes/dot.hpp"     // for NodeDot, dot
#include <algorithm>                  // for equal
#include <cstddef>                    // for size_t
#include <kaad/functions/adjoint.hpp> // for scalar_dot
#include <kaad/functions/primal.hpp>  // for scalar_dot
#include <kaad/graph/graph.hpp>       // for Graph, dot
#include <kaad/graph/node_handle.hpp> // for Node
#include <kaad/graph/nodes/inode.hpp> // for INode
#include <kaad/tensor/tensor.hpp>     // for Tensor
#include <memory>                     // for unique_ptr, make_unique
#include <span>                       // for span
#include <string>                     // for basic_string
#include <utility>                    // for pair
#include <vector>                     // for vector

namespace kaad {

Node dot(Graph &rec, Node lhs, Node rhs) {

    std::size_t rec_len = rec.nodes.size();

    INode *lhs_ptr = rec.get_node(lhs);
    INode *rhs_ptr = rec.get_node(rhs);
    Tensor &lhs_val = lhs_ptr->value();
    Tensor &rhs_val = rhs_ptr->value();

    bool lhs_scalar = lhs_val.size() == 1;
    bool rhs_scalar = rhs_val.size() == 1;

    if (lhs_scalar || rhs_scalar) {

        if (rhs_scalar) {

            rec.nodes.push_back(std::make_unique<NodeDot>(lhs_ptr, rhs_ptr));

        } else if (lhs_scalar) {

            rec.nodes.push_back(std::make_unique<NodeDot>(rhs_ptr, lhs_ptr));
        }

        // override functions to ScalarDot
        static_cast<NodeDot *>(rec.nodes.back().get())->forward_op =
            functions::primal::binary::scalar_dot;
        static_cast<NodeDot *>(rec.nodes.back().get())->backward_op =
            functions::adjoint::binary::scalar_dot;

    } else if (lhs_val.rank() == 1 && rhs_val.rank() == 1 &&
               std::equal(lhs_val.shape().begin(), lhs_val.shape().end(),
                          rhs_val.shape().begin())) {

        rec.nodes.push_back(std::make_unique<NodeDot>(lhs_ptr, rhs_ptr));

    } else {
        throw ShapeError(make_graph_errmsg(
            "shape error", rec_len, "dot",
            "incompatible tensor shapes for dot product",
            {{"A.shape", lhs_val.shape()}, {"B.shape", rhs_val.shape()}}));
    }

    return rec.back_handle();
}

} // namespace kaad
