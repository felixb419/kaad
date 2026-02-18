#include "outer.hpp"

#include "../dispatchers.hpp" // for get_flexOp, get_flexGrad

namespace kaad {

void Node_outer_metadata(Node_outer &node) {
    // compute metadata
    Tensor &lhs = node.lhs->value;
    Tensor &rhs = node.rhs->value;
    Tensor &C = node.value;

    node.C_rank = C.rank();

    node.lhs_stride.resize(node.C_rank);
    node.rhs_stride.resize(node.C_rank);
    node.strideC.resize(node.C_rank);

    std::copy(C.stride().begin(), C.stride().end(), node.strideC.data());
    std::copy(lhs.stride().begin(), lhs.stride().end(), node.lhs_stride.data());
    std::copy(rhs.stride().begin(), rhs.stride().end(),
              node.rhs_stride.data() + lhs.rank());

    node.C_offset.resize(node.C_rank);
    for (int i = 0; i < node.C_rank; i++) {
        node.C_offset[i] = C.shape()[i] * node.strideC[i];
    }

    // assign compile-time recursive function
    if (node.C_rank <= Dispatchers::MAX_NDIMS) {
        node.forward_op =
            Dispatchers::get_flexOp<Node_outer::Kernel>()[node.C_rank];
        node.backward_op =
            Dispatchers::get_flexGrad<Node_outer::Kernel>()[node.C_rank];
    }
}

const char *Node_outer::node_type() const noexcept { return "Node_outer"; }

Node_outer::Node_outer(INode *lhs_ptr, INode *rhs_ptr,
                       std::span<const int> value_shape)
    : lhs(lhs_ptr), rhs(rhs_ptr), INode(value_shape, false) {

    Node_outer_metadata(*this);
}

void Node_outer::eval() {
    if (!this->evaluated) {
        this->lhs->eval();
        this->rhs->eval();

        forward_op(this->lhs->value.data(), this->rhs->value.data(),
                   this->value.elements_.data(), lhs_stride.data(),
                   rhs_stride.data(), strideC.data(), C_offset.data(), C_rank);
        this->evaluated = true;
    }
}

void Node_outer::getGrad() {
    backward_op(this->lhs->value.data(), this->lhs->gradient.elements_.data(),
                this->rhs->value.data(), this->rhs->gradient.elements_.data(),
                this->value.data(), this->gradient.data(), lhs_stride.data(),
                rhs_stride.data(), strideC.data(), C_offset.data(), C_rank);

    if (this->lhs->hasInputs) {
        this->lhs->getGrad();
    }
    if (this->rhs->hasInputs) {
        this->rhs->getGrad();
    }
}

} // namespace kaad
