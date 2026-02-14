#include "inode.hpp"

namespace kaad {

INode::INode(INode *A_ptr, std::span<const int> value_shape)
    : A(A_ptr), evaluated(false), value(value_shape),
      hasInputs(this->A != nullptr), gradient(value) {
    if (this->hasInputs) {
        std::fill(value.elements_.begin(), value.elements_.end(), 0);
    }
    std::fill(gradient.elements_.begin(), gradient.elements_.end(), 0);
}

INode::INode(INode *A_ptr, std::span<const int> value_shape,
             std::span<const int> value_stride)
    : A(A_ptr), evaluated(false), value(value_shape, value_stride),
      hasInputs(this->A != nullptr), gradient(value) {
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
