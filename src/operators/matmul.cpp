#include <kaad/operators/operators.hpp> // for matmul

#include "../graph/common.hpp"             // for combine_matrix
#include "../graph/nodes/batch_matmul.hpp" // for NodeBatchMatmul
#include "../graph/nodes/matmul.hpp"       // for NodeMatmul
#include <algorithm>                       // for max
#include <cstddef>                         // for size_t
#include <kaad/exceptions.hpp>             // for ShapeError, make_graph_er...
#include <kaad/graph/graph.hpp>            // for Graph, matmul
#include <kaad/graph/node_handle.hpp>      // for Node
#include <kaad/graph/nodes/inode.hpp>      // for INode
#include <kaad/tensor/tensor.hpp>          // for Tensor
#include <kaad/tensor/tensor_types.hpp>    // for Shape
#include <memory>                          // for unique_ptr, make_unique
#include <span>                            // for span
#include <string>                          // for basic_string
#include <utility>                         // for pair
#include <vector>                          // for vector

namespace kaad {

Node matmul(Graph &rec, Node lhs, Node rhs) {
    std::size_t rec_len = rec.nodes.size();

    INode *lhs_ptr = rec.get_node(lhs);
    INode *rhs_ptr = rec.get_node(rhs);
    Tensor &lhs_val = lhs_ptr->value();
    Tensor &rhs_val = rhs_ptr->value();

    std::size_t new_len = std::max(lhs_val.rank(), rhs_val.rank());
    Shape new_shape(new_len);

    const char *op_name = new_len == 2 ? "matmul" : "batch_matmul";
    if (!detail::combine_matrix(lhs_val.shape().data(), lhs_val.rank(),
                                rhs_val.shape().data(), rhs_val.rank(),
                                new_shape.data(), new_len)) {
        throw ShapeError(make_graph_errmsg(
            "shape error", rec_len, op_name,
            "incompatible tensor shapes for matrix multiplication",
            {{"A.shape", lhs_val.shape()}, {"B.shape", rhs_val.shape()}}));
    }

    if (new_len == 2) {
        rec.nodes.push_back(
            std::make_unique<NodeMatmul>(lhs_ptr, rhs_ptr, new_shape));
    } else {
        rec.nodes.push_back(
            std::make_unique<NodeBatchMatmul>(lhs_ptr, rhs_ptr, new_shape));
    }

    return rec.back_handle();
}

} // namespace kaad
