#include "mean.hpp"

namespace kaad {

const char *Node_mean::node_type() const noexcept { return "Node_mean"; }

Node_mean::Node_mean(INode *A_ptr, std::span<const int> value_shape)
    : INode(A_ptr, value_shape) {
    this->A_end = A_ptr->value.data() + A_ptr->value.size();
    this->dA_end = A_ptr->gradient.data() + A_ptr->gradient.size();
    this->divisor = A_ptr->value.size();
}

void Node_mean::eval() {
    if (!this->evaluated) {
        this->A->eval();

        forward_op(this->A->value.data(), this->value.elements_.data(), A_end,
                   divisor);
        this->evaluated = true;
    }
}

void Node_mean::getGrad() {
    backward_op(this->A->gradient.elements_.data(), this->gradient.data(),
                dA_end, divisor);

    if (this->A->hasInputs) {
        this->A->getGrad();
    }
}

} // namespace kaad
