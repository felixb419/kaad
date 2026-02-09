#include "outer.hpp"

#include "../dispatchers.hpp" // for get_flexOp, get_flexGrad

namespace kaad {

void Node_outer::metadata() {
    // compute metadata
    Tensor_view A = this->A->value.view();
    Tensor_view B = this->B->value.view();
    Tensor_view C = this->value.view();

    this->C_nDims = C.nDims;

    this->strideA.resize(this->C_nDims);
    this->strideB.resize(this->C_nDims);
    this->strideC.resize(this->C_nDims);

    std::copy(C.stride, C.stride + C.nDims, this->strideC.data());
    std::copy(A.stride, A.stride + A.nDims, this->strideA.data());
    std::copy(B.stride, B.stride + B.nDims, this->strideB.data() + A.nDims);

    this->C_offset.resize(this->C_nDims);
    for (int i = 0; i < this->C_nDims; i++) {
        this->C_offset[i] = C.shape[i] * this->strideC[i];
    }

    // assign compile-time recursive function
    if (C_nDims <= Dispatchers::MAX_NDIMS) {
        forward_op = Dispatchers::get_flexOp<Scalar, Kernel>()[C_nDims];
        backward_op = Dispatchers::get_flexGrad<Scalar, Kernel>()[C_nDims];
    }
}

const char *Node_outer::node_type() const noexcept { return "Node_outer"; }

void Node_outer::eval() {
    if (!this->evaluated) {
        this->A->eval();
        this->B->eval();

        forward_op(this->A->value.data(), this->B->value.data(),
                   this->value.elements_.data(), strideA.data(), strideB.data(),
                   strideC.data(), C_offset.data(), C_nDims);
        this->evaluated = true;
    }
}

void Node_outer::getGrad() {
    backward_op(this->A->value.data(), this->A->gradient.elements_.data(),
                this->B->value.data(), this->B->gradient.elements_.data(),
                this->value.data(), this->gradient.data(), strideA.data(),
                strideB.data(), strideC.data(), C_offset.data(), C_nDims);

    if (this->A->hasInputs) {
        this->A->getGrad();
    }
    if (this->B->hasInputs) {
        this->B->getGrad();
    }
}

} // namespace kaad
