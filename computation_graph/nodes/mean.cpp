#include "mean.hpp"

namespace kaad {

const char *Node_mean::node_type() const noexcept { return "Node_mean"; }

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
