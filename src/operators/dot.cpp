#include <kaad/operators/operators.hpp> // for dot

#include <array>                           // for array
#include <kaad/exceptions.hpp>             // for BroadcastError, make_graph...
#include <kaad/graph/graph.hpp>            // for Graph, dot
#include <kaad/graph/inode.hpp>            // for INode
#include <kaad/graph/node_handle.hpp>      // for Node
#include <kaad/graph/operator_node.hpp>    // for OperatorNode
#include <kaad/operations/dot_product.hpp> // for DotProduct
#include <memory>                          // for unique_ptr, make_unique
#include <string>                          // for basic_string
#include <vector>                          // for vector

namespace kaad {

Node dot(Graph &rec, Node lhs, Node rhs) {

    try {

        rec.nodes.push_back(
            std::make_unique<OperatorNode<operations::DotProduct>>(
                std::array{rec.get_node(lhs), rec.get_node(rhs)}));

    } catch (BroadcastError &err) {

        throw BroadcastError(
            make_graph_errmsg(rec.nodes.size(), "dot", err.what()));
    }

    return rec.back_handle();
}

} // namespace kaad
