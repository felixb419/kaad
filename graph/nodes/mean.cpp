#include "mean.hpp"

namespace kaad {

Node_mean::Node_mean(INode *input_ptr, std::span<const int> value_shape)
    : input(input_ptr), INode(value_shape, false) {
    this->input_end = input_ptr->value().data() + input_ptr->value().size();
    this->input_grad_end =
        input_ptr->gradient().data() + input_ptr->gradient().size();
    this->divisor = input_ptr->value().size();
}

const char *Node_mean::node_type() const noexcept { return "Node_mean"; }

void Node_mean::eval() {
    if (!this->evaluated()) {
        this->input->eval();

        forward_op(this->input->value().data(), this->value().elements_.data(),
                   input_end, divisor);
        this->evaluated_ = true;
    }
}

void Node_mean::getGrad() {
    backward_op(this->input->gradient().elements_.data(),
                this->gradient().data(), input_grad_end, divisor);

    if (this->input->hasInputs()) {
        this->input->getGrad();
    }
}

} // namespace kaad
