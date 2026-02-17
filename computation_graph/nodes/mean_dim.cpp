#include "mean_dim.hpp"

#include "../common.hpp"      // for along_dim_metadata_impl
#include "../dispatchers.hpp" // for get_meanDim, get_meanDim_grad

namespace kaad {

void Node_mean_dim_metadata(Node_mean_dim &node, int dim) {

    // compute metadata
    Tensor &A = node.A->value;
    Tensor &C = node.value;
    Tensor &dA = node.A->gradient;

    node.divisor = A.shape()[dim];
    node.C_end = C.data() + C.size();
    node.dA_end = dA.data() + dA.size();

    detail::along_dim_metadata_impl(A, C, dim, node.A_rank, node.A_offset,
                                    node.strideA, node.strideC);

    // assign compile-time recursive function
    size_t a_rank = node.A->value.rank();
    if (a_rank <= Dispatchers::MAX_NDIMS) {
        node.forward_op = Dispatchers::get_meanDim<Scalar>()[a_rank];
        node.backward_op = Dispatchers::get_meanDim_grad<Scalar>()[a_rank];
    }
}

const char *Node_mean_dim::node_type() const noexcept {
    return "Node_mean_dim";
}

Node_mean_dim::Node_mean_dim(INode *A_ptr, int dim,
                             std::span<const int> value_shape)
    : A(A_ptr), INode(value_shape, false) {

    Node_mean_dim_metadata(*this, dim);
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
