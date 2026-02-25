#include "../../include/kaad/graph/nodes/inode.hpp"
#include "../../include/kaad/exceptions.hpp"    // for argument_error
#include "../../include/kaad/tensor/tensor.hpp" // for Tensor
#include <algorithm>                            // for fill
#include <vector>                               // for vector<>::iterator

namespace kaad {

INode::INode(std::span<const int> value_shape, bool is_input_node,
             std::span<const int> value_stride)
    : evaluated_(is_input_node), hasInputs_(!is_input_node) {

    // if @p value_stride is given its used to construct value_
    if (value_stride.size() != 0) {
        if (value_shape.size() != value_stride.size()) {
            throw argument_error(
                "sizes of shape_value and shape_stride dont match");
        }

        this->value_ = Tensor(value_shape, value_stride);
    } else {
        this->value_ = Tensor(value_shape);
    }

    this->gradient_ = Tensor(this->value_);

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
