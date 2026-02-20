#include "operators.hpp"

#include "../../exceptions.hpp"      // for shape_error
#include "../../tensor/tensor.hpp"   // for Tensor
#include "../common.hpp"             // for combine_matrix
#include "../computation_graph.hpp"  // for Computation_graph
#include "../node_handle.hpp"        // for Node_handle
#include "../nodes/batch_matmul.hpp" // for Node_batch_matmul
#include "../nodes/matmul.hpp"       // for Node_matmul
#include <memory>                    // for std::make_unique

namespace kaad {

Node_handle matmul(Computation_graph &rec, Node_handle A, Node_handle B) {
    int recLen = rec.nodes.size();

    INode *A_ptr = rec.get_node(A);
    INode *B_ptr = rec.get_node(B);
    Tensor &A_val = A_ptr->value();
    Tensor &B_val = B_ptr->value();

    std::size_t newLen = std::max(A_val.rank(), B_val.rank());
    std::vector<int> newShape(newLen);

    const char *opName = newLen == 2 ? "matmul" : "batch_matmul";
    if (!detail::combine_matrix(A_val.shape().data(), A_val.rank(),
                                B_val.shape().data(), B_val.rank(),
                                newShape.data(), newLen)) {
        throw shape_error(make_graph_errmsg(
            "shape error", recLen, opName,
            "incompatible tensor shapes for matrix multiplication",
            {{"A.shape", A_val.shape()}, {"B.shape", B_val.shape()}}));
    }

    if (newLen == 2) {
        rec.nodes.push_back(
            std::move(std::make_unique<Node_matmul>(A_ptr, B_ptr, newShape)));
    } else {
        rec.nodes.push_back(std::move(
            std::make_unique<Node_batch_matmul>(A_ptr, B_ptr, newShape)));
    }

    return rec.back_handle();
}

} // namespace kaad
