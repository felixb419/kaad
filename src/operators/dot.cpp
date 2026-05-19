#include "../graph/operator_node.hpp"
#include "../operations/dot_product.hpp"

#include <array>
#include <kaad/exceptions.hpp>
#include <kaad/graph/graph.hpp>
#include <kaad/graph/node_handle.hpp>
#include <kaad/operators/operators.hpp>
#include <memory>
#include <vector>

namespace kaad {

Node dot(Graph &rec, Node lhs, Node rhs) {

    try {

        rec.nodes.push_back(
            std::make_unique<OperatorNode<operations::DotProduct>>(
                std::array{rec.get_node(lhs), rec.get_node(rhs)}));

    } catch (ShapeError &err) {

        throw ShapeError(
            make_graph_errmsg(rec.nodes.size(), "dot", err.what()));
    }

    return rec.back_handle();
}

} // namespace kaad
