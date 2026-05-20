#include "../operations/matmul.hpp"

#include "../graph/operator_node.hpp"

#include <array>
#include <kaad/exceptions.hpp>
#include <kaad/graph/graph.hpp>
#include <kaad/graph/node_handle.hpp>
#include <kaad/operators/operators.hpp>
#include <vector>

namespace kaad {

Node matmul(Graph &graph, Node lhs, Node rhs) {

    try {

        graph.nodes.push_back(new OperatorNode<operations::Matmul>(
            std::array{graph.get_node(lhs), graph.get_node(rhs)}));

    } catch (BroadcastError &err) {

        throw BroadcastError(
            make_graph_errmsg(graph.nodes.size(), "matmul", err.what()));
    }

    return graph.back_handle();
}

} // namespace kaad
