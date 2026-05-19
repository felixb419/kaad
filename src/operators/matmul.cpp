#include <kaad/operators/operators.hpp>

#include "../graph/operator_node.hpp"
#include "../operations/matmul.hpp"

#include <array>
#include <kaad/exceptions.hpp>
#include <kaad/graph/graph.hpp>
#include <kaad/graph/node_handle.hpp>
#include <memory>
#include <vector>

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
