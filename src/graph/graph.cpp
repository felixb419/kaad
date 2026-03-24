#include <kaad/graph/graph.hpp>

#include <algorithm>                  // for fill
#include <cstddef>                    // for size_t
#include <kaad/exceptions.hpp>        // for ArgumentError
#include <kaad/graph/node_handle.hpp> // for Node
#include <kaad/graph/nodes/inode.hpp> // for INode
#include <kaad/graph/nodes/input.hpp> // for NodeInput
#include <kaad/tensor/tensor.hpp>     // for Tensor
#include <memory>                     // for unique_ptr, make_unique
#include <string>                     // for operator+, to_string, basic_st...
#include <vector>                     // for vector

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

Node Graph::add_input_node(std::span<const int> value_shape,
                           const char *label) {
    this->nodes.push_back(std::make_unique<NodeInput>(value_shape, label));

    return Node(this->nodes.size() - 1, this);
}

std::vector<const Tensor *> Graph::evaluate(std::span<const Node> nodes) {

    std::vector<const Tensor *> values(nodes.size());

    for (std::size_t i = 0; i < nodes.size(); i++) {
        INode *node_ptr = this->get_node(nodes[i]);
        node_ptr->eval();
        values[i] = &node_ptr->value();
    }

    return values;
}

void Graph::reset() {
    for (std::unique_ptr<INode> &node : this->nodes) {
        (*node).reset();
    }
}

std::vector<const Tensor *> Graph::get_gradient(Node output,
                                                std::span<const Node> inputs) {

    INode *output_node = this->get_node(output);
    std::fill(output_node->gradient().elements_.begin(),
              output_node->gradient().elements_.end(), 1.0);

    output_node->get_grad();

    std::vector<const Tensor *> partials(inputs.size());

    for (std::size_t i = 0; i < inputs.size(); i++) {
        partials[i] = &this->get_node(inputs[i])->gradient();
    }

    return partials;
}

} // namespace kaad
