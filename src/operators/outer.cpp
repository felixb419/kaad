#include "../operations/outer.hpp"

#include "../graph/operator_node.hpp"

#include <array>
#include <kaad/graph/graph.hpp>
#include <kaad/graph/node_handle.hpp>
#include <kaad/operators/operators.hpp>
#include <memory>
#include <vector>

namespace kaad {

Node outer(Graph &graph, Node lhs, Node rhs) {

    graph.nodes.push_back(
        std::make_unique<OperatorNode<operations::OuterProduct>>(
            std::array{graph.get_node(lhs), graph.get_node(rhs)}));

    return graph.back_handle();
}

} // namespace kaad
