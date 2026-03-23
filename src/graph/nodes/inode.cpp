#include <kaad/graph/nodes/inode.hpp>

#include <algorithm>              // for __fill_fn, fill
#include <kaad/exceptions.hpp>    // for argument_error
#include <kaad/tensor/tensor.hpp> // for Tensor
#include <vector>                 // for vector

namespace kaad {

INode::INode(std::span<const int> value_shape, bool is_input_node,
             const char *label, std::span<const int> value_stride)
    : gradient_(value_shape), label_(label), evaluated_(is_input_node),
      is_input_node_(is_input_node) {

    if (this->value_.empty() || this->gradient_.empty()) {
        throw argument_error("given value_shape results in tensor with no "
                             "elements which is not allowed");
    }

    // if @p value_stride is given its used to construct value_
    if (!value_stride.empty()) {
        this->value_ = Tensor(value_shape, value_stride);
    } else {
        this->value_ = Tensor(value_shape);
    }
}

Tensor &INode::value() noexcept { return this->value_; }

const Tensor &INode::value() const noexcept { return this->value_; }

Tensor &INode::gradient() noexcept { return this->gradient_; }

const Tensor &INode::gradient() const noexcept { return this->gradient_; }

bool INode::evaluated() const noexcept { return this->evaluated_; }

bool INode::isInput() const noexcept { return this->is_input_node_; }

void INode::reset() {
    if (!this->is_input_node_) {
        std::ranges::fill(value_.elements_, 0);
        evaluated_ = false;
    }
    std::ranges::fill(gradient_.elements_, 0);
}

} // namespace kaad
