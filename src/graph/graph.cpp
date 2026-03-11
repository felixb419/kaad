#include "../../include/kaad/graph/graph.hpp"
#include "../../include/kaad/exceptions.hpp"        // for argument_error
#include "../../include/kaad/graph/node_handle.hpp" // for Node, operator<<
#include "../../include/kaad/graph/nodes/inode.hpp" // for INode
#include "../../include/kaad/graph/nodes/input.hpp" // for Node_input
#include "../../include/kaad/tensor/tensor.hpp"     // for operator<<, Tensor
#include <memory>      // for unique_ptr, make_unique, allocator_t...
#include <ostream>     // for operator<<, basic_ostream, basic_ost...
#include <string>      // for operator+, to_string, char_traits
#include <type_traits> // for add_const_t
#include <utility>     // for as_const
#include <vector>      // for vector

namespace kaad {

Node Graph::back_handle() noexcept {
    return Node(this->nodes.size() - 1, this);
}

INode *Graph::get_node(Node node) {
    if (node.origin_ != this) {
        throw argument_error("node does not belong to this instance of Graph");
    }
    if (node.idx_ >= this->nodes.size()) {
        throw argument_error(std::to_string(node.idx_) +
                             "is not a valid index for this Graph");
    }

    return this->nodes[node.idx_].get();
}

Node Graph::add_input_node(std::span<const int> value_shape,
                           const char *label) {
    this->nodes.push_back(std::make_unique<Node_input>(value_shape, label));

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

std::vector<const Tensor *> Graph::getGradient(Node output,
                                               std::span<const Node> inputs) {

    INode *output_node = this->get_node(output);
    std::fill(output_node->gradient().elements_.begin(),
              output_node->gradient().elements_.end(), 1.0);

    output_node->getGrad();

    std::vector<const Tensor *> partials(inputs.size());

    for (std::size_t i = 0; i < inputs.size(); i++) {
        partials[i] = &this->get_node(inputs[i])->gradient();
    }

    return partials;
}

std::ostream &operator<<(std::ostream &stream, Node node) {
    INode *node_ptr = node.origin_->get_node(node);
    stream << node_ptr->node_type() << " at idx " << node.idx()
           << " of Graph at " << node.origin() << "\n"
           << "value:\n"
           << node_ptr->value() << "\ngradient:\n"
           << node_ptr->gradient();
    return stream;
}

} // namespace kaad
