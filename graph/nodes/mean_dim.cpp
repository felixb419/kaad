#include "mean_dim.hpp"
#include "../../functions/adjoint.hpp" // for mean_dim_fn
#include "../../functions/primal.hpp"  // for mean_dim_fn
#include "../../tensor/tensor.hpp"     // for Tensor
#include "../common.hpp"               // for along_dim_m...
#include "../dispatchers.hpp"          // for get_meanDim
#include <array>                       // for array

namespace kaad {

void Node_mean_dim::metadata(int dim) {

    // compute metadata
    Tensor &input = this->input->value();
    Tensor &value = this->value();
    Tensor &input_grad = this->input->gradient();

    this->divisor = input.shape()[dim];
    this->value_end = value.data() + value.size();
    this->input_grad_end = input_grad.data() + input_grad.size();

    detail::along_dim_metadata_impl(input, value, dim, this->input_rank,
                                    this->input_offset, this->input_stride,
                                    this->value_stride);

    // assign compile-time recursive function
    std::size_t input_rank = input.rank();
    if (input_rank <= Dispatchers::MAX_NDIMS) {
        this->forward_op = Dispatchers::get_meanDim<Scalar>()[input_rank];
        this->backward_op = Dispatchers::get_meanDim_grad<Scalar>()[input_rank];
    }
}

Node_mean_dim::Node_mean_dim(INode *input_ptr, int dim,
                             std::span<const int> value_shape)
    : input(input_ptr), INode(value_shape, false) {

    this->metadata(dim);
}

const char *Node_mean_dim::node_type() const noexcept {
    return "Node_mean_dim";
}

void Node_mean_dim::eval() {
    if (!this->evaluated()) {
        this->input->eval();

        forward_op(this->input->value().data(), this->value().elements_.data(),
                   input_stride.data(), value_stride.data(),
                   input_offset.data(), input_rank, divisor, value_end);
        this->evaluated_ = true;
    }
}

void Node_mean_dim::getGrad() {
    backward_op(this->input->value().data(),
                this->input->gradient().elements_.data(), this->value().data(),
                this->gradient().data(), input_stride.data(),
                value_stride.data(), input_offset.data(), input_rank, divisor,
                input_grad_end);

    if (this->input->hasInputs()) {
        this->input->getGrad();
    }
}

} // namespace kaad
