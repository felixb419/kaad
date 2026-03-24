#include <kaad/graph/nodes/matmul.hpp>

#include <algorithm>                   // for reverse_copy
#include <array>                       // for array
#include <kaad/graph/nodes/inode.hpp>  // for INode
#include <kaad/tensor/tensor.hpp>      // for Tensor
#include <kaad/tensor/tensor_view.hpp> // for TensorView

namespace kaad {

void metadata_impl(TensorViewConst lhs, TensorViewConst rhs,
                   TensorViewConst value, int &a_dim, int &b_dim,
                   int &shared_dim, int *lhs_stride, int *rhs_stride,
                   int *value_stride) {
    a_dim = lhs.shape[0];
    b_dim = rhs.shape[1];
    shared_dim = lhs.shape[1];

    std::ranges::copy(lhs.stride, lhs_stride);
    std::ranges::copy(rhs.stride, rhs_stride);
    std::ranges::copy(value.stride, value_stride);

    int idx;
    int value_idx;
    int value_offset = 0;
    int value_prev;
    for (int i = 1; i <= 2; i++) {
        idx = 2 - i;

        value_idx = static_cast<int>(value.rank()) - i;
        value_prev = value_offset;
        value_offset += ((value_idx >= 0 ? value.shape[value_idx] : i) - 1) *
                        value_stride[idx];
        value_stride[idx] -=
            value_prev + (value_idx + 1 < 2 ? value_stride[value_idx + 1] : 0);
    }
}

void NodeMatmul::metadata() {
    // compute metadata
    TensorViewConst lhs = this->lhs->value().view();
    TensorViewConst rhs = this->rhs->value().view();
    TensorViewConst value = this->value().view();

    std::array<int, 2> lhs_T_shape;
    std::array<int, 2> lhs_T_stride;
    std::ranges::reverse_copy(lhs.shape, lhs_T_shape.begin());
    std::ranges::reverse_copy(lhs.stride, lhs_T_stride.begin());

    TensorViewConst lhs_T = lhs;
    lhs_T.shape = std::span<const int>(lhs_T_shape);
    lhs_T.stride = std::span<const int>(lhs_T_stride);

    std::array<int, 2> rhs_T_shape;
    std::array<int, 2> rhs_T_stride;
    std::ranges::reverse_copy(rhs.shape, rhs_T_shape.begin());
    std::ranges::reverse_copy(rhs.stride, rhs_T_stride.begin());

    TensorViewConst rhs_T = rhs;
    rhs_T.shape = std::span<const int>(rhs_T_shape);
    rhs_T.stride = std::span<const int>(rhs_T_stride);

    metadata_impl(lhs, rhs, value, this->lhs_rows[0], this->rhs_cols[0],
                  this->shared_dim[0], this->lhs_stride.data(),
                  this->rhs_stride.data(), this->value_stride.data());
    // NOLINTNEXTLINE(readability-suspicious-call-argument)
    metadata_impl(value, rhs_T, lhs, this->lhs_rows[1], this->rhs_cols[1],
                  this->shared_dim[1], this->value_stride.data() + 2,
                  this->rhs_stride.data() + 2, this->lhs_stride.data() + 2);
    // NOLINTNEXTLINE(readability-suspicious-call-argument)
    metadata_impl(lhs_T, value, rhs, this->lhs_rows[2], this->rhs_cols[2],
                  this->shared_dim[2], this->lhs_stride.data() + 4,
                  this->value_stride.data() + 4, this->rhs_stride.data() + 4);
}

NodeMatmul::NodeMatmul(INode *lhs_ptr, INode *rhs_ptr,
                       std::span<const int> value_shape)
    : INode(value_shape, false), lhs(lhs_ptr), rhs(rhs_ptr) {

    this->metadata();
}

const char *NodeMatmul::node_type() const noexcept { return "NodeMatmul"; }

void NodeMatmul::eval() {
    if (!this->evaluated()) {
        this->lhs->eval();
        this->rhs->eval();

        forward_op(this->lhs->value().data(), this->rhs->value().data(),
                   this->value().data(), lhs_rows[0], rhs_cols[0],
                   shared_dim[0], lhs_stride.data(), rhs_stride.data(),
                   value_stride.data());
        this->evaluated_ = true;
    }
}

void NodeMatmul::get_grad() {
    backward_op(this->lhs->value().data(), this->lhs->gradient().data(),
                this->rhs->value().data(), this->rhs->gradient().data(),
                this->gradient().data(), lhs_rows.data() + 1,
                rhs_cols.data() + 1, shared_dim.data() + 1,
                lhs_stride.data() + 2, rhs_stride.data() + 2,
                value_stride.data() + 2);

    if (!this->lhs->is_input()) {
        this->lhs->get_grad();
    }
    if (!this->rhs->is_input()) {
        this->rhs->get_grad();
    }
}

} // namespace kaad
