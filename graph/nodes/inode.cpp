#include "inode.hpp"

namespace kaad {

INode::INode(std::span<const int> value_shape, bool is_input_node)
    : evaluated_(false), value_(value_shape), gradient_(value_) {
    if (this->hasInputs_) {
        std::fill(value_.elements_.begin(), value_.elements_.end(), 0);
    }
    std::fill(gradient_.elements_.begin(), gradient_.elements_.end(), 0);
}

INode::INode(std::span<const int> value_shape,
             std::span<const int> value_stride, bool is_input_node)
    : evaluated_(false), value_(value_shape, value_stride), gradient_(value_) {
    if (this->hasInputs_) {
        std::fill(value_.elements_.begin(), value_.elements_.end(), 0);
    }
    std::fill(gradient_.elements_.begin(), gradient_.elements_.end(), 0);
}

Tensor &INode::value() noexcept { return this->value_; }

const Tensor &INode::value() const noexcept { return this->value_; }

Tensor &INode::gradient() noexcept { return this->gradient_; }

const Tensor &INode::gradient() const noexcept { return this->gradient_; }

bool INode::evaluated() const noexcept { return this->evaluated_; }

bool INode::hasInputs() const noexcept { return this->hasInputs_; }

void INode::reset() {
    if (hasInputs_) {
        std::fill(value_.elements_.begin(), value_.elements_.end(), 0);
        evaluated_ = false;
    }
    std::fill(gradient_.elements_.begin(), gradient_.elements_.end(), 0);
}

} // namespace kaad
