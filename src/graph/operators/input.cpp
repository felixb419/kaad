#include "../../graph/input_node.hpp"
#include "kaad/tensor/internal/tensor_types.hpp"

#include <kaad/graph/graph.hpp>
#include <kaad/graph/node_handle.hpp>

namespace kaad {

Node Graph::input(ShapeView shape) {

    this->nodes.push_back(new InputNode(shape));

    return Node(this->nodes.size() - 1, this);
}

} // namespace kaad
