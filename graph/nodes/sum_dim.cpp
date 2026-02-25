#include "sum_dim.hpp"
#include "../../functions/adjoint.hpp" // for sum_dim_fn
#include "../../functions/primal.hpp"  // for sum_dim_fn
#include "../../scalar.hpp"            // for Scalar
#include "../../tensor/tensor.hpp"     // for Tensor
#include "../common.hpp"               // for along_dim_m...
#include "../dispatchers.hpp"          // for get_sumDim
#include <array>                       // for array

namespace kaad {

void Node_sum_dim::metadata(int dim) {
    // compute metadata
    Tensor &input = this->input->value();
    Tensor &value = this->value();

    detail::along_dim_metadata_impl(input, value, dim, this->value_rank,
                                    this->input_offset, this->input_stride,
                                    this->value_stride);

    // assign compile-time recursive function
    std::size_t input_rank = input.rank();
    if (input_rank <= Dispatchers::MAX_NDIMS) {
        this->val_func = Dispatchers::get_sumDim<Scalar>()[input_rank];
        this->grad_func = Dispatchers::get_sumDim_grad<Scalar>()[input_rank];
    }
}

Node_sum_dim::Node_sum_dim(INode *input_ptr, int dim,
                           std::span<const int> value_shape)
    : input(input_ptr), INode(value_shape, false) {

    this->metadata(dim);
}

const char *Node_sum_dim::node_type() const noexcept { return "Node_sum_dim"; }

void Node_sum_dim::eval() {
    if (!this->evaluated()) {
        this->input->eval();

        val_func(this->input->value().data(), this->value().elements_.data(),
                 input_stride.data(), value_stride.data(), input_offset.data(),
                 value_rank);
        this->evaluated_ = true;
    }
}

void Node_sum_dim::getGrad() {
    grad_func(this->input->gradient().elements_.data(), this->gradient().data(),
              input_stride.data(), value_stride.data(), input_offset.data(),
              value_rank);

    if (this->input->hasInputs()) {
        this->input->getGrad();
    }
}

} // namespace kaad
