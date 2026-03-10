#include "../../../include/kaad/graph/nodes/matmul.hpp"
#include "../../../include/kaad/tensor/tensor.hpp"      // for Tensor
#include "../../../include/kaad/tensor/tensor_view.hpp" // for Tensor_view
#include <algorithm>                                    // for reverse_copy
#include <vector>                                       // for vector

namespace kaad {

void metadata_impl(const Tensor_view lhs, const Tensor_view rhs,
                   const Tensor_view value, int &a_dim, int &b_dim,
                   int &shared_dim, int *lhs_stride, int *rhs_stride,
                   int *value_stride) {
    a_dim = lhs.shape[0];
    b_dim = rhs.shape[1];
    shared_dim = lhs.shape[1];

    std::copy(lhs.stride, lhs.stride + 2, lhs_stride);
    std::copy(rhs.stride, rhs.stride + 2, rhs_stride);
    std::copy(value.stride, value.stride + 2, value_stride);

    int idx;
    int value_idx;
    int value_offset = 0;
    int value_prev;
    for (int i = 1; i <= 2; i++) {
        idx = 2 - i;

        value_idx = static_cast<int>(value.rank) - i;
        value_prev = value_offset;
        value_offset += ((value_idx >= 0 ? value.shape[value_idx] : i) - 1) *
                        value_stride[idx];
        value_stride[idx] -=
            value_prev + (value_idx + 1 < 2 ? value_stride[value_idx + 1] : 0);
    }
}

void Node_matmul::metadata() {
    // compute metadata
    Tensor_view lhs = this->lhs->value().view();
    Tensor_view rhs = this->rhs->value().view();
    Tensor_view value = this->value().view();

    int lhs_T_shape[2];
    int lhs_T_stride[2];
    std::reverse_copy(lhs.shape, lhs.shape + lhs.rank, lhs_T_shape);
    std::reverse_copy(lhs.stride, lhs.stride + lhs.rank, lhs_T_stride);

    Tensor_view lhs_T = lhs;
    lhs_T.shape = lhs_T_shape;
    lhs_T.stride = lhs_T_stride;

    int rhs_T_shape[2];
    int rhs_T_stride[2];
    std::reverse_copy(rhs.shape, rhs.shape + rhs.rank, rhs_T_shape);
    std::reverse_copy(rhs.stride, rhs.stride + rhs.rank, rhs_T_stride);

    Tensor_view rhs_T = rhs;
    rhs_T.shape = rhs_T_shape;
    rhs_T.stride = rhs_T_stride;

    metadata_impl(lhs, rhs, value, this->lhs_rows[0], this->rhs_cols[0],
                  this->shared_dim[0], this->lhs_stride, this->rhs_stride,
                  this->value_stride);
    metadata_impl(value, rhs_T, lhs, this->lhs_rows[1], this->rhs_cols[1],
                  this->shared_dim[1], this->value_stride + 2,
                  this->rhs_stride + 2, this->lhs_stride + 2);
    metadata_impl(lhs_T, value, rhs, this->lhs_rows[2], this->rhs_cols[2],
                  this->shared_dim[2], this->lhs_stride + 4,
                  this->value_stride + 4, this->rhs_stride + 4);
}

Node_matmul::Node_matmul(INode *lhs_ptr, INode *rhs_ptr,
                         std::span<const int> value_shape)
    : INode(value_shape, false), lhs(lhs_ptr), rhs(rhs_ptr) {

    this->metadata();
}

const char *Node_matmul::node_type() const noexcept { return "Node_matmul"; }

void Node_matmul::eval() {
    if (!this->evaluated()) {
        this->lhs->eval();
        this->rhs->eval();

        forward_op(this->lhs->value().data(), this->rhs->value().data(),
                   this->value().data(), lhs_rows[0], rhs_cols[0],
                   shared_dim[0], lhs_stride, rhs_stride, value_stride);
        this->evaluated_ = true;
    }
}

void Node_matmul::getGrad() {
    backward_op(this->lhs->value().data(), this->lhs->gradient().data(),
                this->rhs->value().data(), this->rhs->gradient().data(),
                this->gradient().data(), lhs_rows + 1, rhs_cols + 1,
                shared_dim + 1, lhs_stride + 2, rhs_stride + 2,
                value_stride + 2);

    if (!this->lhs->isInput()) {
        this->lhs->getGrad();
    }
    if (!this->rhs->isInput()) {
        this->rhs->getGrad();
    }
}

} // namespace kaad
