#include "outer.hpp"

#include "../dispatchers.hpp" // for get_flexOp, get_flexGrad

namespace kaad {

void Node_outer::metadata() {
    // compute metadata
    Tensor_view A = this->A->value.view();
    Tensor_view B = this->B->value.view();
    Tensor_view C = this->value.view();

    this->C_rank = C.rank;

    this->strideA.resize(this->C_rank);
    this->strideB.resize(this->C_rank);
    this->strideC.resize(this->C_rank);

    std::copy(C.stride, C.stride + C.rank, this->strideC.data());
    std::copy(A.stride, A.stride + A.rank, this->strideA.data());
    std::copy(B.stride, B.stride + B.rank, this->strideB.data() + A.rank);

    this->C_offset.resize(this->C_rank);
    for (int i = 0; i < this->C_rank; i++) {
        this->C_offset[i] = C.shape[i] * this->strideC[i];
    }

    // assign compile-time recursive function
    if (C_rank <= Dispatchers::MAX_NDIMS) {
        forward_op = Dispatchers::get_flexOp<Scalar, Kernel>()[C_rank];
        backward_op = Dispatchers::get_flexGrad<Scalar, Kernel>()[C_rank];
    }
}

const char *Node_outer::node_type() const noexcept { return "Node_outer"; }

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
