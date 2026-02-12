#include "sum_dim.hpp"

namespace kaad {

void Node_sum_dim::metadata(int dim) {
    // compute metadata
    Tensor &A = this->A->value;
    Tensor &C = this->value;

    detail::along_dim_metadata_impl(A, C, dim, this->C_rank, this->A_offset,
                                    this->strideA, this->strideC);

    // assign compile-time recursive function
    size_t a_rank = static_cast<INode *>(this)->A->value.rank();
    if (a_rank <= Dispatchers::MAX_NDIMS) {
        val_func = Dispatchers::get_sumDim<Scalar>()[a_rank];
        grad_func = Dispatchers::get_sumDim_grad<Scalar>()[a_rank];
    }
}

const char *Node_sum_dim::node_type() const noexcept { return "Node_sum_dim"; }

void Node_sum_dim::eval() {
    if (!this->evaluated) {
        this->A->eval();

        val_func(this->A->value.data(), this->value.elements_.data(),
                 strideA.data(), strideC.data(), A_offset.data(), C_rank);
        this->evaluated = true;
    }
}

void Node_sum_dim::getGrad() {
    grad_func(this->A->gradient.elements_.data(), this->gradient.data(),
              strideA.data(), strideC.data(), A_offset.data(), C_rank);

    if (this->A->hasInputs) {
        this->A->getGrad();
    }
}

} // namespace kaad
