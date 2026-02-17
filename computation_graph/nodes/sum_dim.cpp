#include "sum_dim.hpp"

namespace kaad {

void Node_sum_dim_metadata(Node_sum_dim &node, int dim) {
    // compute metadata
    Tensor &A = node.A->value;
    Tensor &C = node.value;

    detail::along_dim_metadata_impl(A, C, dim, node.C_rank, node.A_offset,
                                    node.strideA, node.strideC);

    // assign compile-time recursive function
    size_t a_rank = static_cast<INode *>(&node)->A->value.rank();
    if (a_rank <= Dispatchers::MAX_NDIMS) {
        node.val_func = Dispatchers::get_sumDim<Scalar>()[a_rank];
        node.grad_func = Dispatchers::get_sumDim_grad<Scalar>()[a_rank];
    }
}

const char *Node_sum_dim::node_type() const noexcept { return "Node_sum_dim"; }

Node_sum_dim::Node_sum_dim(INode *A_ptr, int dim,
                           std::span<const int> value_shape)
    : INode(A_ptr, value_shape) {
    Node_sum_dim_metadata(*this, dim);
}

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
