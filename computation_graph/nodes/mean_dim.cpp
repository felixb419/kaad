#include "mean_dim.hpp"

#include "../common.hpp"      // for along_dim_metadata_impl
#include "../dispatchers.hpp" // for get_meanDim, get_meanDim_grad

namespace kaad {

void Node_mean_dim::metadata(int dim) {

    // compute metadata
    Tensor &A = this->A->value;
    Tensor &C = this->value;
    Tensor &dA = this->A->gradient;

    this->divisor = A.shape()[dim];
    this->C_end = C.data() + C.size();
    this->dA_end = dA.data() + dA.size();

    detail::along_dim_metadata_impl(A, C, dim, this->A_rank, this->A_offset,
                                    this->strideA, this->strideC);

    // assign compile-time recursive function
    size_t a_rank = static_cast<INode *>(this)->A->value.rank();
    if (a_rank <= Dispatchers::MAX_NDIMS) {
        forward_op = Dispatchers::get_meanDim<Scalar>()[a_rank];
        backward_op = Dispatchers::get_meanDim_grad<Scalar>()[a_rank];
    }
}

const char *Node_mean_dim::node_type() const noexcept {
    return "Node_mean_dim";
}

void Node_mean_dim::eval() {
    if (!this->evaluated) {
        this->A->eval();

        forward_op(this->A->value.data(), this->value.elements_.data(),
                   strideA.data(), strideC.data(), A_offset.data(), A_rank,
                   divisor, C_end);
        this->evaluated = true;
    }
}

void Node_mean_dim::getGrad() {
    backward_op(this->A->value.data(), this->A->gradient.elements_.data(),
                this->value.data(), this->gradient.data(), strideA.data(),
                strideC.data(), A_offset.data(), A_rank, divisor, dA_end);

    if (this->A->hasInputs) {
        this->A->getGrad();
    }
}

} // namespace kaad
