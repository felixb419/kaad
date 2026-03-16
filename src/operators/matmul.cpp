#include <kaad/operators/operators.hpp> // for matmul

#include <algorithm>                         // for max
#include <cstddef>                           // for size_t
#include <kaad/exceptions.hpp>               // for shape_error, make_graph...
#include <kaad/graph/common.hpp>             // for combine_matrix
#include <kaad/graph/graph.hpp>              // for Graph, matmul
#include <kaad/graph/node_handle.hpp>        // for Node
#include <kaad/graph/nodes/batch_matmul.hpp> // for Node_batch_matmul
#include <kaad/graph/nodes/inode.hpp>        // for INode
#include <kaad/graph/nodes/matmul.hpp>       // for Node_matmul
#include <kaad/tensor/tensor.hpp>            // for Tensor
#include <memory>                            // for unique_ptr, make_unique
#include <span>                              // for span
#include <string>                            // for basic_string
#include <utility>                           // for pair
#include <vector>                            // for vector

namespace kaad {

Node matmul(Graph &rec, Node lhs, Node rhs) {
    std::size_t recLen = rec.nodes.size();

    INode *lhs_ptr = rec.get_node(lhs);
    INode *rhs_ptr = rec.get_node(rhs);
    Tensor &lhs_val = lhs_ptr->value();
    Tensor &rhs_val = rhs_ptr->value();

    std::size_t newLen = std::max(lhs_val.rank(), rhs_val.rank());
    std::vector<int> newShape(newLen);

    const char *opName = newLen == 2 ? "matmul" : "batch_matmul";
    if (!detail::combine_matrix(lhs_val.shape().data(), lhs_val.rank(),
                                rhs_val.shape().data(), rhs_val.rank(),
                                newShape.data(), newLen)) {
        throw shape_error(make_graph_errmsg(
            "shape error", recLen, opName,
            "incompatible tensor shapes for matrix multiplication",
            {{"A.shape", lhs_val.shape()}, {"B.shape", rhs_val.shape()}}));
    }

    if (newLen == 2) {
        rec.nodes.push_back(
            std::make_unique<Node_matmul>(lhs_ptr, rhs_ptr, newShape));
    } else {
        rec.nodes.push_back(
            std::make_unique<Node_batch_matmul>(lhs_ptr, rhs_ptr, newShape));
    }

    return rec.back_handle();
}

} // namespace kaad
