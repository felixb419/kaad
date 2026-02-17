#include "transp.hpp"

namespace kaad {

const char *Node_transp::node_type() const noexcept { return "Node_transp"; }

Node_transp::Node_transp(INode *A_ptr, std::span<const int> value_shape,
                         std::span<const int> value_stride)
    : INode(A_ptr, value_shape, value_stride) {
    INode *base_ptr = static_cast<INode *>(this);
    this->A_end = base_ptr->A->value.data() + base_ptr->A->value.size();
    this->C_end = base_ptr->value.data() + base_ptr->value.size();
}

void Node_transp::eval() {
    if (!this->evaluated) {
        this->A->eval();

        forward_op(this->A->value.data(), this->value.elements_.data(), A_end);
        this->evaluated = true;
    }
}

void Node_transp::getGrad() {
    backward_op(this->A->value.data(), this->A->gradient.elements_.data(),
                this->value.data(), this->gradient.data(), C_end);

    if (this->A->hasInputs) {
        this->A->getGrad();
    }
}

} // namespace kaad
