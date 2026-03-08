#include "../../../include/kaad/graph/nodes/inode.hpp"
#include "../../../include/kaad/exceptions.hpp"    // for argument_error
#include "../../../include/kaad/tensor/tensor.hpp" // for Tensor
#include <algorithm>                               // for fill
#include <vector>                                  // for vector<>::iterator

namespace kaad {

INode::INode(std::span<const int> value_shape, bool is_input_node,
             const char *label, std::span<const int> value_stride)
    : label_(label), evaluated_(is_input_node), is_input_node_(is_input_node) {

    // if @p value_stride is given its used to construct value_
    if (value_stride.size() != 0) {
        this->value_ = Tensor(value_shape, value_stride);
    } else {
        this->value_ = Tensor(value_shape);
    }

    if (!this->is_input_node_) {
        std::fill(value_.elements_.begin(), value_.elements_.end(), 0);
    }

    this->gradient_ = Tensor::zeros(value_shape);
}

Tensor &INode::value() noexcept { return this->value_; }

const Tensor &INode::value() const noexcept { return this->value_; }

Tensor &INode::gradient() noexcept { return this->gradient_; }

const Tensor &INode::gradient() const noexcept { return this->gradient_; }

bool INode::evaluated() const noexcept { return this->evaluated_; }

bool INode::isInput() const noexcept { return this->is_input_node_; }

void INode::reset() {
    if (!this->is_input_node_) {
        std::fill(value_.elements_.begin(), value_.elements_.end(), 0);
        evaluated_ = false;
    }
    std::fill(gradient_.elements_.begin(), gradient_.elements_.end(), 0);
}

} // namespace kaad
