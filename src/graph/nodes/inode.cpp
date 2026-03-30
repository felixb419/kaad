#include <kaad/graph/nodes/inode.hpp>

#include <algorithm>                    // for __fill_fn, fill
#include <kaad/exceptions.hpp>          // for ShapeError, to_string
#include <kaad/tensor/tensor.hpp>       // for Tensor
#include <kaad/tensor/tensor_types.hpp> // for ShapeView, StridesView
#include <string>                       // for char_traits, operator+, basi...
#include <vector>                       // for allocator, vector

namespace kaad {

INode::INode(ShapeView value_shape, bool is_input_node, const char *label,
             StridesView value_strides)
    : gradient_(value_shape), label_(label), evaluated_(is_input_node),
      is_input_node_(is_input_node) {

    if (this->value_.empty() || this->gradient_.empty()) {
        throw ShapeError("given value_shape (" + to_string(value_shape) +
                         ") results in tensor with no "
                         "elements which is not allowed");
    }

    // if @p value_strides is given its used to construct value_
    if (!value_strides.empty()) {
        this->value_ = Tensor(value_shape, value_strides);
    } else {
        this->value_ = Tensor(value_shape);
    }
}

Tensor &INode::value() noexcept { return this->value_; }

const Tensor &INode::value() const noexcept { return this->value_; }

Tensor &INode::gradient() noexcept { return this->gradient_; }

const Tensor &INode::gradient() const noexcept { return this->gradient_; }

bool INode::evaluated() const noexcept { return this->evaluated_; }

bool INode::is_input() const noexcept { return this->is_input_node_; }

void INode::reset() {
    if (!this->is_input_node_) {
        std::ranges::fill(value_.elements_, 0);
        evaluated_ = false;
    }
    std::ranges::fill(gradient_.elements_, 0);
}

} // namespace kaad
