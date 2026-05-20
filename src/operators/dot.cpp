#include "../graph/operator_node.hpp"
#include "../operations/dot_product.hpp"

#include <array>
#include <kaad/exceptions.hpp>
#include <kaad/graph/graph.hpp>
#include <kaad/graph/node_handle.hpp>
#include <kaad/operators/operators.hpp>
#include <vector>

namespace kaad {

Node dot(Graph &graph, Node lhs, Node rhs) {

    try {

        graph.nodes.push_back(new OperatorNode<operations::DotProduct>(
            std::array{graph.get_node(lhs), graph.get_node(rhs)}));

    } catch (ShapeError &err) {

        throw ShapeError(
            make_graph_errmsg(graph.nodes.size(), "dot", err.what()));
    }

    return graph.back_handle();
}

} // namespace kaad
