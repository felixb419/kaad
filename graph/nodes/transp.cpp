#include "transp.hpp"

namespace kaad {

const char *Node_transp::node_type() const noexcept { return "Node_transp"; }

Node_transp::Node_transp(INode *input_ptr, std::span<const int> value_shape,
                         std::span<const int> value_stride)
    : input(input_ptr), INode(value_shape, value_stride, false) {
    this->input_end = this->input->value().data() + this->input->value().size();
    this->value_end = this->value().data() + this->value().size();
}

void Node_transp::eval() {
    if (!this->evaluated()) {
        this->input->eval();

        forward_op(this->input->value().data(), this->value().elements_.data(),
                   input_end);
        this->evaluated_ = true;
    }
}

void Node_transp::getGrad() {
    backward_op(this->input->value().data(),
                this->input->gradient().elements_.data(), this->value().data(),
                this->gradient().data(), value_end);

    if (this->input->hasInputs()) {
        this->input->getGrad();
    }
}

} // namespace kaad
