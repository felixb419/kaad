#include <kaad/operators/operators.hpp> // for dot

#include "../graph/nodes/dot.hpp"         // for NodeDot, dot
#include <cstddef>                        // for size_t
#include <kaad/exceptions.hpp>            // for BroadcastError, make_graph...
#include <kaad/functions/dot_product.hpp> // for DotProduct
#include <kaad/graph/graph.hpp>           // for Graph, dot
#include <kaad/graph/node_handle.hpp>     // for Node
#include <kaad/graph/nodes/inode.hpp>     // for INode
#include <memory>                         // for unique_ptr, make_unique
#include <string>                         // for basic_string
#include <vector>                         // for vector

namespace kaad {

Node dot(Graph &rec, Node lhs, Node rhs) {

    std::size_t rec_len = rec.nodes.size();

    try {

        functions::DotProduct::broadcast(lhs.shape(), rhs.shape());

    } catch (BroadcastError &err) {

        throw BroadcastError(make_graph_errmsg(rec_len, "dot", err.what()));
    }

    rec.nodes.push_back(
        std::make_unique<NodeDot>(rec.get_node(lhs), rec.get_node(rhs)));

    return rec.back_handle();
}

} // namespace kaad
