#include <kaad/operators/operators.hpp> // for matmul

#include "../graph/operator_node.hpp"    // for OperatorNode
#include "../operations/matmul.hpp"      // for Matmul
#include <array>                         // for array
#include <kaad/exceptions.hpp>           // for BroadcastError, make_graph_e...
#include <kaad/graph/graph.hpp>          // for Graph, matmul
#include <kaad/graph/internal/inode.hpp> // for INode
#include <kaad/graph/node_handle.hpp>    // for Node
#include <memory>                        // for unique_ptr, make_unique
#include <string>                        // for basic_string
#include <vector>                        // for vector

namespace kaad {

Node matmul(Graph &rec, Node lhs, Node rhs) {

    try {

        rec.nodes.push_back(std::make_unique<OperatorNode<operations::Matmul>>(
            std::array{rec.get_node(lhs), rec.get_node(rhs)}));

    } catch (BroadcastError &err) {

        throw BroadcastError(
            make_graph_errmsg(rec.nodes.size(), "matmul", err.what()));
    }

    return rec.back_handle();
}

} // namespace kaad
