#include "slice.hpp"

#include <algorithm>                    // for copy
#include <array>                        // for array
#include <kaad/graph/dispatchers.hpp>   // for get_slice, get_slice_grad
#include <kaad/graph/nodes/inode.hpp>   // for INode
#include <kaad/max_rank.hpp>            // for KAAD_MAX_RANK
#include <kaad/scalar.hpp>              // for Scalar
#include <kaad/static_vector.hpp>       // for StaticVector
#include <kaad/tensor/tensor.hpp>       // for Tensor
#include <kaad/tensor/tensor_types.hpp> // for ShapeView

namespace kaad {

void NodeSlice::metadata(const int *offset_arr) {
    // compute metadata
    Tensor &input = this->input->value();
    Tensor &value = this->value();

    this->value_rank = value.rank();
    this->input_strides.resize(this->value_rank);
    this->value_strides.resize(this->value_rank);

    int idx;
    int input_idx;
    int value_dix;
    for (std::size_t i = 1; i <= this->value_rank; i++) {
        idx = static_cast<int>(this->value_rank - i);
        input_idx = static_cast<int>(input.rank() - i);
        this->input_strides[idx] =
            input_idx >= 0 ? input.strides()[input_idx] : 0;
        value_dix = static_cast<int>(value.rank() - i);
        this->value_strides[idx] =
            value_dix >= 0 ? value.strides()[value_dix] : 0;
        // make sure value_strides[idx] is 1 instead of 0 if value.shape[idx] is
        // 1 for traversing in flexible function
        if (this->value_strides[idx] == 0 && value.shape()[value_dix] == 1) {
            this->value_strides[idx] = 1;
        }
    }

    this->value_offset.resize(this->value_rank);
    for (std::size_t i = 0; i < this->value_rank; i++) {
        this->value_offset[i] =
            static_cast<int>(value.shape()[i] * this->value_strides[i]);
    }

    this->start_offset_a.resize(input.rank());
    std::copy(offset_arr, offset_arr + input.rank(),
              this->start_offset_a.data());
    for (std::size_t i = 0; i < input.rank(); i++) {
        this->start_offset_a[i] *= this->input_strides[i];
    }

    // assign compile-time recursive function
    std::size_t a_rank = this->input->value().rank();
    if (a_rank < KAAD_MAX_RANK) {
        this->forward_op = Dispatchers::get_slice<Scalar>()[a_rank];
        this->backward_op = Dispatchers::get_slice_grad<Scalar>()[a_rank];
    }
}

NodeSlice::NodeSlice(INode *input_ptr, const int *offset_arr,
                     ShapeView value_shape)
    : INode(value_shape, false), input(input_ptr) {
    this->metadata(offset_arr);
}

const char *NodeSlice::node_type() const noexcept { return "NodeSlice"; }

void NodeSlice::eval() {
    if (!this->evaluated()) {
        this->input->eval();

        forward_op(this->input->value().data(), this->value().data(),
                   input_strides.data(), value_strides.data(),
                   start_offset_a.data(), value_offset.data(), value_rank);
        this->evaluated_ = true;
    }
}

void NodeSlice::get_grad() {
    backward_op(this->input->gradient().data(), this->gradient().data(),
                input_strides.data(), value_strides.data(),
                start_offset_a.data(), value_offset.data(), value_rank);

    if (!this->input->is_input()) {
        this->input->get_grad();
    }
}

} // namespace kaad
