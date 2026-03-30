#include <kaad/operators/operators.hpp> // for dot

#include "../graph/nodes/dot.hpp"         // for NodeDot, dot
#include <cstddef>                        // for size_t
#include <kaad/exceptions.hpp>            // for BroadcastError, make_graph...
#include <kaad/functions/dot_product.hpp> // for DotProduct
#include <kaad/graph/graph.hpp>           // for Graph, dot
#include <kaad/graph/node_handle.hpp>     // for Node
#include <kaad/graph/nodes/inode.hpp>     // for INode
#include <kaad/tensor/tensor.hpp>         // for Tensor
#include <kaad/tensor/tensor_types.hpp>   // for Shape
#include <memory>                         // for unique_ptr, make_unique
#include <span>                           // for span
#include <string>                         // for basic_string
#include <utility>                        // for pair
#include <vector>                         // for vector

namespace kaad {

Node dot(Graph &rec, Node lhs, Node rhs) {

    std::size_t rec_len = rec.nodes.size();

    INode *lhs_ptr = rec.get_node(lhs);
    INode *rhs_ptr = rec.get_node(rhs);
    Tensor &lhs_val = lhs_ptr->value();
    Tensor &rhs_val = rhs_ptr->value();

    Shape new_shape; // not actually needed since it will always be {1}.

    if (functions::DotProduct::broadcast(lhs_val.shape(), rhs_val.shape(),
                                         new_shape)) {

        rec.nodes.push_back(std::make_unique<NodeDot>(lhs_ptr, rhs_ptr));

    } else {
        throw BroadcastError(make_graph_errmsg(
            rec_len, "dot", "incompatible tensor shapes for dot product",
            {{"A.shape", lhs_val.shape()}, {"B.shape", rhs_val.shape()}}));
    }

    return rec.back_handle();
}

} // namespace kaad
