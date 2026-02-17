#include "slice.hpp"

#include "../dispatchers.hpp" // for get_slice, get_slice_grad

namespace kaad {

void Node_slice_metadata(Node_slice &node, const int *offset_arr) {
    // compute metadata
    Tensor &A = node.A->value;
    Tensor &C = node.value;

    node.C_rank = C.rank();
    node.strideA.resize(node.C_rank);
    node.strideC.resize(node.C_rank);

    int idx, idxA, idxC;
    for (int i = 1; i <= node.C_rank; i++) {
        idx = node.C_rank - i;
        idxA = A.rank() - i;
        node.strideA[idx] = idxA >= 0 ? A.stride()[idxA] : 0;
        idxC = C.rank() - i;
        node.strideC[idx] = idxC >= 0 ? C.stride()[idxC] : 0;
        // make sure strideC[idx] is 1 instead of 0 if C.shape[idx] is 1 for
        // traversing in flexible function
        if (node.strideC[idx] == 0 && C.shape()[idxC] == 1) {
            node.strideC[idx] = 1;
        }
    }

    node.C_offset.resize(node.C_rank);
    for (int i = 0; i < node.C_rank; i++) {
        node.C_offset[i] = C.shape()[i] * node.strideC[i];
    }

    node.start_offset_a.resize(A.rank());
    std::copy(offset_arr, offset_arr + A.rank(), node.start_offset_a.data());
    for (int i = 0; i < A.rank(); i++) {
        node.start_offset_a[i] *= node.strideA[i];
    }

    // assign compile-time recursive function
    size_t a_rank = node.A->value.rank();
    if (a_rank < Dispatchers::MAX_NDIMS) {
        node.forward_op = Dispatchers::get_slice<Scalar>()[a_rank];
        node.backward_op = Dispatchers::get_slice_grad<Scalar>()[a_rank];
    }
}

const char *Node_slice::node_type() const noexcept { return "Node_slice"; }

Node_slice::Node_slice(INode *A_ptr, const int *offset_arr,
                       std::span<const int> value_shape)
    : A(A_ptr), INode(value_shape, false) {
    Node_slice_metadata(*this, offset_arr);
}

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
