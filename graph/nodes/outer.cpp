#include "outer.hpp"

#include "../dispatchers.hpp" // for get_flexOp, get_flexGrad

namespace kaad {

void Node_outer::metadata() {
    // compute metadata
    Tensor &lhs = this->lhs->value();
    Tensor &rhs = this->rhs->value();
    Tensor &C = this->value();

    this->C_rank = C.rank();

    this->lhs_stride.resize(this->C_rank);
    this->rhs_stride.resize(this->C_rank);
    this->strideC.resize(this->C_rank);

    std::copy(C.stride().begin(), C.stride().end(), this->strideC.data());
    std::copy(lhs.stride().begin(), lhs.stride().end(),
              this->lhs_stride.data());
    std::copy(rhs.stride().begin(), rhs.stride().end(),
              this->rhs_stride.data() + lhs.rank());

    this->C_offset.resize(this->C_rank);
    for (int i = 0; i < this->C_rank; i++) {
        this->C_offset[i] = C.shape()[i] * this->strideC[i];
    }

    // assign compile-time recursive function
    if (this->C_rank <= Dispatchers::MAX_NDIMS) {
        this->forward_op =
            Dispatchers::get_flexOp<Node_outer::Kernel>()[this->C_rank];
        this->backward_op =
            Dispatchers::get_flexGrad<Node_outer::Kernel>()[this->C_rank];
    }
}

const char *Node_outer::node_type() const noexcept { return "Node_outer"; }

Node_outer::Node_outer(INode *lhs_ptr, INode *rhs_ptr,
                       std::span<const int> value_shape)
    : lhs(lhs_ptr), rhs(rhs_ptr), INode(value_shape, false) {

    this->metadata();
}

void Node_outer::eval() {
    if (!this->evaluated()) {
        this->lhs->eval();
        this->rhs->eval();

        forward_op(this->lhs->value().data(), this->rhs->value().data(),
                   this->value().elements_.data(), lhs_stride.data(),
                   rhs_stride.data(), strideC.data(), C_offset.data(), C_rank);
        this->evaluated_ = true;
    }
}

void Node_outer::getGrad() {
    backward_op(
        this->lhs->value().data(), this->lhs->gradient().elements_.data(),
        this->rhs->value().data(), this->rhs->gradient().elements_.data(),
        this->value().data(), this->gradient().data(), lhs_stride.data(),
        rhs_stride.data(), strideC.data(), C_offset.data(), C_rank);

    if (this->lhs->hasInputs()) {
        this->lhs->getGrad();
    }
    if (this->rhs->hasInputs()) {
        this->rhs->getGrad();
    }
}

} // namespace kaad
