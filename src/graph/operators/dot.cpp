#include "../../graph/operator_node.hpp"
#include "../../operations/dot_product.hpp"

#include <array>
#include <kaad/exceptions.hpp>
#include <kaad/graph/graph.hpp>
#include <kaad/graph/node_handle.hpp>
#include <vector>

namespace kaad {

Node Graph::dot(Node lhs, Node rhs) {

    try {

        this->nodes.push_back(new OperatorNode<operations::DotProduct>(
            std::array{this->get_node(lhs), this->get_node(rhs)}));

    } catch (ShapeError &err) {

        throw ShapeError(
            make_graph_errmsg(this->nodes.size(), "dot", err.what()));
    }

    return Node(this->nodes.size() - 1, this);
}

} // namespace kaad
