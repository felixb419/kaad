#include "sum_dim.hpp"

namespace kaad {

void Node_sum_dim_metadata(Node_sum_dim &node, int dim) {
    // compute metadata
    Tensor &input = node.input->value;
    Tensor &value = node.value;

    detail::along_dim_metadata_impl(input, value, dim, node.value_rank,
                                    node.input_offset, node.input_stride,
                                    node.value_stride);

    // assign compile-time recursive function
    std::size_t a_rank = node.input->value.rank();
    if (a_rank <= Dispatchers::MAX_NDIMS) {
        node.val_func = Dispatchers::get_sumDim<Scalar>()[a_rank];
        node.grad_func = Dispatchers::get_sumDim_grad<Scalar>()[a_rank];
    }
}

const char *Node_sum_dim::node_type() const noexcept { return "Node_sum_dim"; }

Node_sum_dim::Node_sum_dim(INode *input_ptr, int dim,
                           std::span<const int> value_shape)
    : input(input_ptr), INode(value_shape, false) {
    Node_sum_dim_metadata(*this, dim);
}

void Node_sum_dim::eval() {
    if (!this->evaluated) {
        this->input->eval();

        val_func(this->input->value.data(), this->value.elements_.data(),
                 input_stride.data(), value_stride.data(), input_offset.data(),
                 value_rank);
        this->evaluated = true;
    }
}

void Node_sum_dim::getGrad() {
    grad_func(this->input->gradient.elements_.data(), this->gradient.data(),
              input_stride.data(), value_stride.data(), input_offset.data(),
              value_rank);

    if (this->input->hasInputs) {
        this->input->getGrad();
    }
}

} // namespace kaad
