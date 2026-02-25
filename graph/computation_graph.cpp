#include "../include/kaad/graph/computation_graph.hpp"
#include "../include/kaad/exceptions.hpp" // for argument_error
#include "../include/kaad/graph/computation_graph.hpp"
#include "../include/kaad/graph/node_handle.hpp" // for Node_handle, operator<<
#include "../include/kaad/graph/nodes/inode.hpp" // for INode
#include "../include/kaad/graph/nodes/input.hpp" // for Node_input
#include "../include/kaad/tensor/tensor.hpp"     // for operator<<, Tensor
#include <memory>      // for unique_ptr, make_unique, allocator_t...
#include <ostream>     // for operator<<, basic_ostream, basic_ost...
#include <string>      // for operator+, to_string, char_traits
#include <type_traits> // for add_const_t
#include <utility>     // for as_const
#include <vector>      // for vector

namespace kaad {

Node_handle Computation_graph::back_handle() noexcept {
    return Node_handle(this->nodes.size() - 1, this);
}

INode *Computation_graph::get_node(Node_handle node) {
    return const_cast<INode *>(std::as_const(*this).get_node(node));
}

const INode *Computation_graph::get_node(Node_handle node) const {
    if (node.origin_ != this) {
        throw argument_error(
            "node does not belong to this instance of Computation_graph");
    }
    if (node.idx_ < 0 || node.idx_ >= this->nodes.size()) {
        throw argument_error(std::to_string(node.idx_) +
                             "is not a valid index for this Computation_graph");
    }

    return this->nodes[node.idx_].get();
}

Node_handle
Computation_graph::add_input_node(std::span<const int> value_shape,
                                  std::span<Scalar> &node_value_elements) {
    this->nodes.push_back(std::make_unique<Node_input>(value_shape));

    Tensor &Node_value = nodes.back()->value();
    node_value_elements = std::span<Scalar>(
        Node_value.data(), Node_value.data() + Node_value.size());

    return Node_handle(this->nodes.size() - 1, this);
}

void Computation_graph::reset() {
    for (int i = 0; i < nodes.size(); i++) {
        nodes[i]->reset();
    }
}

std::ostream &operator<<(std::ostream &os, Node_handle node) {
    INode *node_ptr = node.origin_->get_node(node);
    os << node_ptr->node_type() << " at idx " << node.idx()
       << " of Computation_graph at " << node.origin() << "\n"
       << "value:\n"
       << node_ptr->value() << "\ngradient:\n"
       << node_ptr->gradient();
    return os;
}

} // namespace kaad
