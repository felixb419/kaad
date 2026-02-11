#include "dot.hpp"

namespace kaad {

const char *Node_dot::node_type() const noexcept { return "Node_dot"; }

Node_dot::Node_dot(INode *A_ptr, INode *B_ptr)
    : B(B_ptr), INode(A_ptr, Scalar(0)) {

    INode *base_ptr = static_cast<INode *>(this);
    this->A_end = base_ptr->A->value.data() + base_ptr->A->value.size();
}

void Node_dot::eval() {
    if (!this->evaluated) {
        this->A->eval();
        this->B->eval();

        forward_op(this->A->value.data(), this->B->value.data(),
                   this->value.elements_.data(), A_end);
        this->evaluated = true;
    }
}

void Node_dot::getGrad() {
    backward_op(this->A->value.data(), this->A->gradient.elements_.data(),
                this->B->value.data(), this->B->gradient.elements_.data(),
                this->value.data(), this->gradient.data(), A_end);

    if (this->A->hasInputs) {
        this->A->getGrad();
    }
    if (this->B->hasInputs) {
        this->B->getGrad();
    }
}

} // namespace kaad
