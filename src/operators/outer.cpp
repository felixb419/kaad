#include <kaad/operators/operators.hpp>

#include "../graph/operator_node.hpp"
#include "../operations/outer.hpp"
#include <array>
#include <kaad/graph/graph.hpp>
#include <kaad/graph/node_handle.hpp>
#include <memory>
#include <vector>

namespace kaad {

Node outer(Graph &rec, Node lhs, Node rhs) {

    rec.nodes.push_back(
        std::make_unique<OperatorNode<operations::OuterProduct>>(
            std::array{rec.get_node(lhs), rec.get_node(rhs)}));

    return rec.back_handle();
}

} // namespace kaad
