#include "outer.hpp"

#include "../dispatchers.hpp" // for get_flexOp, get_flexGrad

namespace kaad {

void Node_outer_metadata(Node_outer &node) {
    // compute metadata
    Tensor &A = node.A->value;
    Tensor &B = node.B->value;
    Tensor &C = node.value;

    node.C_rank = C.rank();

    node.strideA.resize(node.C_rank);
    node.strideB.resize(node.C_rank);
    node.strideC.resize(node.C_rank);

    std::copy(C.stride().begin(), C.stride().end(), node.strideC.data());
    std::copy(A.stride().begin(), A.stride().end(), node.strideA.data());
    std::copy(B.stride().begin(), B.stride().end(),
              node.strideB.data() + A.rank());

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

Node_outer::Node_outer(INode *A_ptr, INode *B_ptr,
                       std::span<const int> value_shape)
    : A(A_ptr), B(B_ptr), INode(value_shape, false) {

    Node_outer_metadata(*this);
}

void Node_outer::eval() {
    if (!this->evaluated) {
        this->A->eval();
        this->B->eval();

        forward_op(this->A->value.data(), this->B->value.data(),
                   this->value.elements_.data(), strideA.data(), strideB.data(),
                   strideC.data(), C_offset.data(), C_rank);
        this->evaluated = true;
    }
}

void Node_outer::getGrad() {
    backward_op(this->A->value.data(), this->A->gradient.elements_.data(),
                this->B->value.data(), this->B->gradient.elements_.data(),
                this->value.data(), this->gradient.data(), strideA.data(),
                strideB.data(), strideC.data(), C_offset.data(), C_rank);

    if (this->A->hasInputs) {
        this->A->getGrad();
    }
    if (this->B->hasInputs) {
        this->B->getGrad();
    }
}

} // namespace kaad
