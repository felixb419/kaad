#include "slice.hpp"

#include "../dispatchers.hpp" // for get_slice, get_slice_grad

namespace kaad {

void Node_slice::metadata(const int *offset_arr) {
    // compute metadata
    Tensor_view A = this->A->value.view();
    Tensor_view C = this->value.view();

    this->C_rank = C.rank;
    this->strideA.resize(this->C_rank);
    this->strideC.resize(this->C_rank);

    int idx, idxA, idxC;
    for (int i = 1; i <= this->C_rank; i++) {
        idx = this->C_rank - i;
        idxA = A.rank - i;
        this->strideA[idx] = idxA >= 0 ? A.stride[idxA] : 0;
        idxC = C.rank - i;
        this->strideC[idx] = idxC >= 0 ? C.stride[idxC] : 0;
        // make sure strideC[idx] is 1 instead of 0 if C.shape[idx] is 1 for
        // traversing in flexible function
        if (this->strideC[idx] == 0 && C.shape[idxC] == 1) {
            this->strideC[idx] = 1;
        }
    }

    this->C_offset.resize(this->C_rank);
    for (int i = 0; i < this->C_rank; i++) {
        this->C_offset[i] = C.shape[i] * this->strideC[i];
    }

    this->start_offset_a.resize(A.rank);
    std::copy(offset_arr, offset_arr + A.rank, this->start_offset_a.data());
    for (int i = 0; i < A.rank; i++) {
        this->start_offset_a[i] *= this->strideA[i];
    }

    // assign compile-time recursive function
    size_t a_rank = static_cast<INode *>(this)->A->value.rank();
    if (a_rank < Dispatchers::MAX_NDIMS) {
        forward_op = Dispatchers::get_slice<Scalar>()[a_rank];
        backward_op = Dispatchers::get_slice_grad<Scalar>()[a_rank];
    }
}

const char *Node_slice::node_type() const noexcept { return "Node_slice"; }

void Node_slice::eval() {
    if (!this->evaluated) {
        this->A->eval();

        forward_op(this->A->value.data(), this->value.elements_.data(),
                   strideA.data(), strideC.data(), start_offset_a.data(),
                   C_offset.data(), C_rank);
        this->evaluated = true;
    }
}

void Node_slice::getGrad() {
    backward_op(this->A->gradient.elements_.data(), this->gradient.data(),
                strideA.data(), strideC.data(), start_offset_a.data(),
                C_offset.data(), C_rank);

    if (this->A->hasInputs) {
        this->A->getGrad();
    }
}

} // namespace kaad
