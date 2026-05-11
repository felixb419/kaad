#include <kaad/graph/graph.hpp>

#include "input_node.hpp"                        // for InputNode
#include <algorithm>                             // for __fill_fn, fill
#include <cstddef>                               // for size_t
#include <kaad/exceptions.hpp>                   // for ArgumentError
#include <kaad/graph/internal/inode.hpp>         // for INode
#include <kaad/graph/node_handle.hpp>            // for Node
#include <kaad/tensor/internal/tensor_types.hpp> // for ShapeView
#include <kaad/tensor/tensor_view.hpp> // for TensorViewConst, TensorView
#include <memory>                      // for unique_ptr, make_unique
#include <span>
#include <string>                      // for basic_string, operator+, to_...
#include <vector>                      // for vector

namespace kaad {

Node Graph::back_handle() noexcept {
    return Node(this->nodes.size() - 1, this);
}

INode *Graph::get_node(Node node) {
    if (node.origin_ != this) {
        throw ArgumentError("node does not belong to this instance of Graph");
    }
    if (node.idx_ >= this->nodes.size()) {
        throw ArgumentError(std::to_string(node.idx_) +
                            "is not a valid index for this Graph");
    }

    return this->nodes[node.idx_].get();
}

Node Graph::add_input_node(ShapeView value_shape) {
    this->nodes.push_back(std::make_unique<InputNode>(value_shape));

    return Node(this->nodes.size() - 1, this);
}

std::vector<TensorViewConst> Graph::evaluate(std::span<const Node> nodes) {

    std::vector<TensorViewConst> values(nodes.size());

    for (std::size_t i = 0; i < nodes.size(); i++) {
        INode *node_ptr = this->get_node(nodes[i]);
        node_ptr->evaluate();
        values[i] = node_ptr->value();
    }

    return values;
}

void Graph::reset() {
    for (std::unique_ptr<INode> &node : this->nodes) {
        (*node).reset();
    }
}

std::vector<TensorViewConst> Graph::get_gradient(Node output,
                                                 std::span<const Node> inputs) {

    INode *output_node = this->get_node(output);
    std::ranges::fill(output_node->gradient_mut().elements, 1.0);

    output_node->acc_input_gradients();

    std::vector<TensorViewConst> partials(inputs.size());

    for (std::size_t i = 0; i < inputs.size(); i++) {
        partials[i] = this->get_node(inputs[i])->gradient();
    }

    return partials;
}

} // namespace kaad
