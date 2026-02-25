#include "../nodes/dot.hpp"            // for Node_dot, dot
#include "../../exceptions.hpp"        // for make_graph_errmsg
#include "../../functions/adjoint.hpp" // for scalarDot
#include "../../functions/primal.hpp"  // for scalarDot
#include "../../graph/nodes/inode.hpp" // for INode
#include "../../tensor/tensor.hpp"     // for Tensor
#include "../computation_graph.hpp"    // for Computation_graph
#include "../node_handle.hpp"          // for Node_handle
#include "operators.hpp"               // for dot
#include <algorithm>                   // for equal
#include <memory>                      // for unique_ptr, mak...
#include <utility>                     // for move
#include <vector>                      // for vector

namespace kaad {

Node_handle dot(Computation_graph &rec, Node_handle A, Node_handle B) {

    int recLen = rec.nodes.size();

    INode *A_ptr = rec.get_node(A);
    INode *B_ptr = rec.get_node(B);
    Tensor &A_val = A_ptr->value();
    Tensor &B_val = B_ptr->value();

    bool A_scalar = A_val.rank() == 1 && A_val.shape()[0] == 1;
    bool B_scalar = B_val.rank() == 1 && B_val.shape()[0] == 1;

    if (A_scalar || B_scalar) {

        if (B_scalar) {

            rec.nodes.push_back(
                std::move(std::make_unique<Node_dot>(A_ptr, B_ptr)));

        } else if (A_scalar) {

            rec.nodes.push_back(
                std::move(std::make_unique<Node_dot>(B_ptr, A_ptr)));
        }

        // override functions to ScalarDot
        static_cast<Node_dot *>(rec.nodes.back().get())->forward_op =
            functions::primal::binary::scalarDot;
        static_cast<Node_dot *>(rec.nodes.back().get())->backward_op =
            functions::adjoint::binary::scalarDot;

    } else if (A_val.rank() == 1 && B_val.rank() == 1 &&
               std::equal(A_val.shape().begin(), A_val.shape().end(),
                          B_val.shape().begin())) {

        rec.nodes.push_back(
            std::move(std::make_unique<Node_dot>(A_ptr, B_ptr)));

    } else {
        throw shape_error(make_graph_errmsg(
            "shape error", recLen, "dot",
            "incompatible tensor shapes for dot product",
            {{"A.shape", A_val.shape()}, {"B.shape", B_val.shape()}}));
    }

    return rec.back_handle();
}

} // namespace kaad
