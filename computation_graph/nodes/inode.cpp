#include "inode.hpp"

namespace kaad {

INode::INode(std::span<const int> value_shape, bool is_input_node)
    : evaluated(false), value(value_shape), gradient(value) {
    if (this->hasInputs) {
        std::fill(value.elements_.begin(), value.elements_.end(), 0);
    }
    std::fill(gradient.elements_.begin(), gradient.elements_.end(), 0);
}

INode::INode(std::span<const int> value_shape,
             std::span<const int> value_stride, bool is_input_node)
    : evaluated(false), value(value_shape, value_stride), gradient(value) {
    if (this->hasInputs) {
        std::fill(value.elements_.begin(), value.elements_.end(), 0);
    }
    std::fill(gradient.elements_.begin(), gradient.elements_.end(), 0);
}

void INode::reset() {
    if (hasInputs) {
        std::fill(value.elements_.begin(), value.elements_.end(), 0);
        evaluated = false;
    }
    std::fill(gradient.elements_.begin(), gradient.elements_.end(), 0);
}

} // namespace kaad
