#include "sum_dim.hpp"

namespace kaad {

void Node_sum_dim::metadata(int dim) {
    // compute metadata
    Tensor_view A = this->A->value.view();
    Tensor_view C = this->value.view();

    detail::along_dim_metadata_impl(A, C, dim, this->C_nDims, this->A_offset,
                                    this->strideA, this->strideC);

    // assign compile-time recursive function
    size_t a_ndims = static_cast<INode *>(this)->A->value.nDims();
    if (a_ndims <= Dispatchers::MAX_NDIMS) {
        val_func = Dispatchers::get_sumDim<Scalar>()[a_ndims];
        grad_func = Dispatchers::get_sumDim_grad<Scalar>()[a_ndims];
    }
}

const char *Node_sum_dim::node_type() const noexcept { return "Node_sum_dim"; }

void Node_sum_dim::eval() {
    if (!this->evaluated) {
        this->A->eval();

        val_func(this->A->value.data(), this->value.elements_.data(),
                 strideA.data(), strideC.data(), A_offset.data(), C_nDims);
        this->evaluated = true;
    }
}

void Node_sum_dim::getGrad() {
    grad_func(this->A->gradient.elements_.data(), this->gradient.data(),
              strideA.data(), strideC.data(), A_offset.data(), C_nDims);

    if (this->A->hasInputs) {
        this->A->getGrad();
    }
}

} // namespace kaad
