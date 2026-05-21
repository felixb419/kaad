#include "../../graph/input_node.hpp"
#include "kaad/tensor/internal/tensor_types.hpp"

#include <kaad/graph/graph.hpp>
#include <kaad/graph/node_handle.hpp>
#include <span>
#include <vector>

namespace kaad {

Node Graph::input(ShapeView shape) {

    this->nodes.push_back(new InputNode(shape));

    return Node(this->nodes.size() - 1, this);
}

std::vector<Node> Graph::input(std::span<ShapeView> shapes) {

    std::vector<Node> nodes;
    nodes.reserve(shapes.size());

    for (ShapeView shape : shapes) {

        this->nodes.push_back(new InputNode(shape));
        nodes.push_back(Node(this->nodes.size() - 1, this));
    }

    return nodes;
}

} // namespace kaad
