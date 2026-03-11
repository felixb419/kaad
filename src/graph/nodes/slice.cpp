#include "../../../include/kaad/graph/nodes/slice.hpp"
#include "../../../include/kaad/graph/dispatchers.hpp" // for get_slice
#include "../../../include/kaad/graph/nodes/inode.hpp" // for INode
#include "../../../include/kaad/scalar.hpp"            // for Scalar
#include "../../../include/kaad/tensor/tensor.hpp"     // for Tensor
#include <algorithm>                                   // for copy
#include <array>                                       // for array

namespace kaad {

void Node_slice::metadata(const int *offset_arr) {
    // compute metadata
    Tensor &input = this->input->value();
    Tensor &value = this->value();

    this->value_rank = value.rank();
    this->input_stride.resize(this->value_rank);
    this->value_stride.resize(this->value_rank);

    int idx;
    int input_idx;
    int value_dix;
    for (std::size_t i = 1; i <= this->value_rank; i++) {
        idx = static_cast<int>(this->value_rank - i);
        input_idx = static_cast<int>(input.rank() - i);
        this->input_stride[idx] =
            input_idx >= 0 ? input.stride()[input_idx] : 0;
        value_dix = static_cast<int>(value.rank() - i);
        this->value_stride[idx] =
            value_dix >= 0 ? value.stride()[value_dix] : 0;
        // make sure value_stride[idx] is 1 instead of 0 if value.shape[idx] is
        // 1 for traversing in flexible function
        if (this->value_stride[idx] == 0 && value.shape()[value_dix] == 1) {
            this->value_stride[idx] = 1;
        }
    }

    this->value_offset.resize(this->value_rank);
    for (std::size_t i = 0; i < this->value_rank; i++) {
        this->value_offset[i] =
            static_cast<int>(value.shape()[i] * this->value_stride[i]);
    }

    this->start_offset_a.resize(input.rank());
    std::copy(offset_arr, offset_arr + input.rank(),
              this->start_offset_a.data());
    for (std::size_t i = 0; i < input.rank(); i++) {
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
    : INode(value_shape, false), input(input_ptr) {
    this->metadata(offset_arr);
}

const char *Node_slice::node_type() const noexcept { return "Node_slice"; }

void Node_slice::eval() {
    if (!this->evaluated()) {
        this->input->eval();

        forward_op(this->input->value().data(), this->value().data(),
                   input_stride.data(), value_stride.data(),
                   start_offset_a.data(), value_offset.data(), value_rank);
        this->evaluated_ = true;
    }
}

void Node_slice::getGrad() {
    backward_op(this->input->gradient().data(), this->gradient().data(),
                input_stride.data(), value_stride.data(), start_offset_a.data(),
                value_offset.data(), value_rank);

    if (!this->input->isInput()) {
        this->input->getGrad();
    }
}

} // namespace kaad
