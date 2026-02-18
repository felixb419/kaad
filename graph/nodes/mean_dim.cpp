#include "mean_dim.hpp"

#include "../common.hpp"      // for along_dim_metadata_impl
#include "../dispatchers.hpp" // for get_meanDim, get_meanDim_grad

namespace kaad {

void Node_mean_dim_metadata(Node_mean_dim &node, int dim) {

    // compute metadata
    Tensor &input = node.input->value;
    Tensor &value = node.value;
    Tensor &input_grad = node.input->gradient;

    node.divisor = input.shape()[dim];
    node.value_end = value.data() + value.size();
    node.input_grad_end = input_grad.data() + input_grad.size();

    detail::along_dim_metadata_impl(input, value, dim, node.input_rank,
                                    node.input_offset, node.input_stride,
                                    node.value_stride);

    // assign compile-time recursive function
    std::size_t a_rank = node.input->value.rank();
    if (a_rank <= Dispatchers::MAX_NDIMS) {
        node.forward_op = Dispatchers::get_meanDim<Scalar>()[a_rank];
        node.backward_op = Dispatchers::get_meanDim_grad<Scalar>()[a_rank];
    }
}

const char *Node_mean_dim::node_type() const noexcept {
    return "Node_mean_dim";
}

Node_mean_dim::Node_mean_dim(INode *input_ptr, int dim,
                             std::span<const int> value_shape)
    : input(input_ptr), INode(value_shape, false) {

    Node_mean_dim_metadata(*this, dim);
}

void Node_mean_dim::eval() {
    if (!this->evaluated) {
        this->input->eval();

        forward_op(this->input->value.data(), this->value.elements_.data(),
                   input_stride.data(), value_stride.data(),
                   input_offset.data(), input_rank, divisor, value_end);
        this->evaluated = true;
    }
}

void Node_mean_dim::getGrad() {
    backward_op(this->input->value.data(),
                this->input->gradient.elements_.data(), this->value.data(),
                this->gradient.data(), input_stride.data(), value_stride.data(),
                input_offset.data(), input_rank, divisor, input_grad_end);

    if (this->input->hasInputs) {
        this->input->getGrad();
    }
}

} // namespace kaad
