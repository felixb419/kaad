#include <kaad/operators/operators.hpp> // for matmul

#include "../graph/nodes/matmul.hpp"    // for NodeMatmul
#include <cstddef>                      // for size_t
#include <kaad/exceptions.hpp>          // for BroadcastError, to_string
#include <kaad/functions/matmul.hpp>    // for Matmul
#include <kaad/graph/graph.hpp>         // for Graph, matmul
#include <kaad/graph/node_handle.hpp>   // for Node
#include <kaad/graph/nodes/inode.hpp>   // for INode
#include <kaad/tensor/tensor.hpp>       // for Tensor
#include <kaad/tensor/tensor_types.hpp> // for Shape
#include <memory>                       // for unique_ptr, allocator, make_...
#include <string>                       // for char_traits, basic_string
#include <vector>                       // for vector

namespace kaad {

Node matmul(Graph &rec, Node lhs, Node rhs) {
    std::size_t rec_len = rec.nodes.size();

    INode *lhs_ptr = rec.get_node(lhs);
    INode *rhs_ptr = rec.get_node(rhs);
    Tensor &lhs_val = lhs_ptr->value();
    Tensor &rhs_val = rhs_ptr->value();

    Shape new_shape;
    if (!functions::Matmul::broadcast(lhs_val.shape(), rhs_val.shape(),
                                      new_shape)) {
        throw BroadcastError(
            make_graph_errmsg(rec_len, "matmul",
                              "incompatible tensor shapes for matrix "
                              "multiplication, lhs.shape()" +
                                  to_string(lhs_val.shape()) + ", rhs.shape()" +
                                  to_string(rhs_val.shape())));
    }

    rec.nodes.push_back(
        std::make_unique<NodeMatmul>(lhs_ptr, rhs_ptr, new_shape));

    return rec.back_handle();
}

} // namespace kaad
