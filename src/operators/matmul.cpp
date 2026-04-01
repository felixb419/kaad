#include <kaad/operators/operators.hpp> // for matmul

#include "../graph/nodes/matmul.hpp"    // for NodeMatmul
#include <cstddef>                      // for size_t
#include <kaad/exceptions.hpp>          // for BroadcastError, make_graph_e...
#include <kaad/functions/matmul.hpp>    // for Matmul
#include <kaad/graph/graph.hpp>         // for Graph, matmul
#include <kaad/graph/node_handle.hpp>   // for Node
#include <kaad/graph/nodes/inode.hpp>   // for INode
#include <kaad/tensor/tensor_types.hpp> // for Shape
#include <memory>                       // for unique_ptr, make_unique
#include <string>                       // for basic_string
#include <vector>                       // for vector

namespace kaad {

Node matmul(Graph &rec, Node lhs, Node rhs) {
    std::size_t rec_len = rec.nodes.size();

    Shape new_shape;

    try {

        new_shape = functions::Matmul::broadcast(lhs.shape(), rhs.shape());

    } catch (BroadcastError &err) {

        throw BroadcastError(make_graph_errmsg(rec_len, "matmul", err.what()));
    }

    rec.nodes.push_back(std::make_unique<NodeMatmul>(
        rec.get_node(lhs), rec.get_node(rhs), new_shape));

    return rec.back_handle();
}

} // namespace kaad
