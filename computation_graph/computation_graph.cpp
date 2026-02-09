#include "computation_graph.hpp"
#include "../tensor/tensor.hpp" // for Tensor
#include "node_handle.hpp"      // for Node_handle
#include "nodes/inode.hpp"      // for INode
#include <memory>               // for std::unique_ptr, std::make_unique
#include <vector>               // for std::vector

namespace kaad {

INode *Computation_graph::get_node(Node_handle node) {
    if (node.origin_ != this) {
        throw std::invalid_argument(
            "node does not belong to this instance of Computation_graph");
    }
    if (node.idx_ < 0 && node.idx_ >= this->nodes.size()) {
        throw std::invalid_argument(
            std::to_string(node.idx_) +
            "is not a valid index for this Computation_graph");
    }

    return this->nodes[node.idx_].get();
}

Node_handle Computation_graph::back_handle() {
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
       << node_ptr->value << "\ngradient:\n"
       << node_ptr->gradient;
    return os;
}

} // namespace kaad
