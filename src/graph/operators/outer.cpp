#include "../../operations/outer.hpp"

#include "../../graph/operator_node.hpp"

#include <array>
#include <kaad/graph/graph.hpp>
#include <kaad/graph/node_handle.hpp>
#include <vector>

namespace kaad {

Node Graph::outer(Node lhs, Node rhs) {

    this->nodes.push_back(new OperatorNode<operations::OuterProduct>(
        std::array{this->get_node(lhs), this->get_node(rhs)}));

    return Node(this->nodes.size() - 1, this);
}

} // namespace kaad
