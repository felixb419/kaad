#include "slice.hpp"

#include "../dispatchers.hpp" // for get_slice, get_slice_grad

namespace kaad {

void Node_slice_metadata(Node_slice &node, const int *offset_arr) {
    // compute metadata
    Tensor &input = node.input->value;
    Tensor &value = node.value;

    node.value_rank = value.rank();
    node.input_stride.resize(node.value_rank);
    node.value_stride.resize(node.value_rank);

    int idx, input_idx, value_dix;
    for (int i = 1; i <= node.value_rank; i++) {
        idx = node.value_rank - i;
        input_idx = input.rank() - i;
        node.input_stride[idx] = input_idx >= 0 ? input.stride()[input_idx] : 0;
        value_dix = value.rank() - i;
        node.value_stride[idx] = value_dix >= 0 ? value.stride()[value_dix] : 0;
        // make sure value_stride[idx] is 1 instead of 0 if value.shape[idx] is
        // 1 for traversing in flexible function
        if (node.value_stride[idx] == 0 && value.shape()[value_dix] == 1) {
            node.value_stride[idx] = 1;
        }
    }

    node.value_offset.resize(node.value_rank);
    for (int i = 0; i < node.value_rank; i++) {
        node.value_offset[i] = value.shape()[i] * node.value_stride[i];
    }

    node.start_offset_a.resize(input.rank());
    std::copy(offset_arr, offset_arr + input.rank(),
              node.start_offset_a.data());
    for (int i = 0; i < input.rank(); i++) {
        node.start_offset_a[i] *= node.input_stride[i];
    }

    // assign compile-time recursive function
    size_t a_rank = node.input->value.rank();
    if (a_rank < Dispatchers::MAX_NDIMS) {
        node.forward_op = Dispatchers::get_slice<Scalar>()[a_rank];
        node.backward_op = Dispatchers::get_slice_grad<Scalar>()[a_rank];
    }
}

const char *Node_slice::node_type() const noexcept { return "Node_slice"; }

Node_slice::Node_slice(INode *input_ptr, const int *offset_arr,
                       std::span<const int> value_shape)
    : input(input_ptr), INode(value_shape, false) {
    Node_slice_metadata(*this, offset_arr);
}

void Node_slice::eval() {
    if (!this->evaluated) {
        this->input->eval();

        forward_op(this->input->value.data(), this->value.elements_.data(),
                   input_stride.data(), value_stride.data(),
                   start_offset_a.data(), value_offset.data(), value_rank);
        this->evaluated = true;
    }
}

void Node_slice::getGrad() {
    backward_op(this->input->gradient.elements_.data(), this->gradient.data(),
                input_stride.data(), value_stride.data(), start_offset_a.data(),
                value_offset.data(), value_rank);

    if (this->input->hasInputs) {
        this->input->getGrad();
    }
}

} // namespace kaad
