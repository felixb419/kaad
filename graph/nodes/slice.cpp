#include "slice.hpp"
#include "../../functions/adjoint.hpp" // for slice_fn
#include "../../functions/primal.hpp"  // for slice_fn
#include "../../tensor/tensor.hpp"     // for Tensor
#include "../dispatchers.hpp"          // for get_slice
#include <algorithm>                   // for copy
#include <array>                       // for array

namespace kaad {

void Node_slice::metadata(const int *offset_arr) {
    // compute metadata
    Tensor &input = this->input->value();
    Tensor &value = this->value();

    this->value_rank = value.rank();
    this->input_stride.resize(this->value_rank);
    this->value_stride.resize(this->value_rank);

    int idx, input_idx, value_dix;
    for (int i = 1; i <= this->value_rank; i++) {
        idx = this->value_rank - i;
        input_idx = input.rank() - i;
        this->input_stride[idx] =
            input_idx >= 0 ? input.stride()[input_idx] : 0;
        value_dix = value.rank() - i;
        this->value_stride[idx] =
            value_dix >= 0 ? value.stride()[value_dix] : 0;
        // make sure value_stride[idx] is 1 instead of 0 if value.shape[idx] is
        // 1 for traversing in flexible function
        if (this->value_stride[idx] == 0 && value.shape()[value_dix] == 1) {
            this->value_stride[idx] = 1;
        }
    }

    this->value_offset.resize(this->value_rank);
    for (int i = 0; i < this->value_rank; i++) {
        this->value_offset[i] = value.shape()[i] * this->value_stride[i];
    }

    this->start_offset_a.resize(input.rank());
    std::copy(offset_arr, offset_arr + input.rank(),
              this->start_offset_a.data());
    for (int i = 0; i < input.rank(); i++) {
        this->start_offset_a[i] *= this->input_stride[i];
    }

    // assign compile-time recursive function
    std::size_t a_rank = this->input->value().rank();
    if (a_rank < Dispatchers::MAX_NDIMS) {
        this->forward_op = Dispatchers::get_slice<Scalar>()[a_rank];
        this->backward_op = Dispatchers::get_slice_grad<Scalar>()[a_rank];
    }
}

Node_slice::Node_slice(INode *input_ptr, const int *offset_arr,
                       std::span<const int> value_shape)
    : input(input_ptr), INode(value_shape, false) {
    this->metadata(offset_arr);
}

const char *Node_slice::node_type() const noexcept { return "Node_slice"; }

void Node_slice::eval() {
    if (!this->evaluated()) {
        this->input->eval();

        forward_op(this->input->value().data(), this->value().elements_.data(),
                   input_stride.data(), value_stride.data(),
                   start_offset_a.data(), value_offset.data(), value_rank);
        this->evaluated_ = true;
    }
}

void Node_slice::getGrad() {
    backward_op(this->input->gradient().elements_.data(),
                this->gradient().data(), input_stride.data(),
                value_stride.data(), start_offset_a.data(), value_offset.data(),
                value_rank);

    if (this->input->hasInputs()) {
        this->input->getGrad();
    }
}

} // namespace kaad
