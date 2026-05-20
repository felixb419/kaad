#include "../../graph/operator_node.hpp"
#include "../../operations/matmul.hpp"

#include <array>
#include <kaad/exceptions.hpp>
#include <kaad/graph/graph.hpp>
#include <kaad/graph/node_handle.hpp>
#include <vector>

namespace kaad {

Node Graph::matmul(Node lhs, Node rhs) {

    try {

        this->nodes.push_back(new OperatorNode<operations::Matmul>(
            std::array{this->get_node(lhs), this->get_node(rhs)}));

    } catch (BroadcastError &err) {

        throw BroadcastError(
            make_graph_errmsg(this->nodes.size(), "matmul", err.what()));
    }

    return Node(this->nodes.size() - 1, this);
}

} // namespace kaad
