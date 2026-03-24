#include <kaad/graph/nodes/matmul.hpp>

#include "kaad/functions/matmul.hpp"    // for Matmul
#include "kaad/tensor/tensor_types.hpp" // for Shape, Stride
#include <algorithm>                    // for __copy_fn, copy
#include <kaad/graph/nodes/inode.hpp>   // for INode
#include <kaad/tensor/tensor.hpp>       // for Tensor
#include <kaad/tensor/tensor_view.hpp>  // for TensorView, TensorViewConst

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

NodeMatmul::NodeMatmul(INode *lhs_ptr, INode *rhs_ptr,
                       std::span<const int> value_shape)
    : INode(value_shape, false), lhs(lhs_ptr), rhs(rhs_ptr) {

    TensorView lhs_v = this->lhs->value().view();
    TensorView rhs_v = this->rhs->value().view();
    TensorView res_v = this->value().view();

    // make lhs^T
    Shape lhs_shape_buff;
    Stride lhs_stride_buff;
    TensorView lhs_t = lhs_v.transpose_2d(lhs_shape_buff, lhs_stride_buff);

    // make rhs^T
    Shape rhs_shape_buff;
    Stride rhs_stride_buff;
    TensorView rhs_t = rhs_v.transpose_2d(rhs_shape_buff, rhs_stride_buff);

    // Compute metadata for individual passes
    // lhs * rhs = res
    this->forward = functions::Matmul::Metadata(lhs_v, rhs_v, res_v);

    // d_res * rhs^T = d_lhs
    this->backward_wrt_lhs = functions::Matmul::Metadata(res_v, rhs_t, lhs_v);

    // lhs^T * d_res = d_rhs
    this->backward_wrt_rhs = functions::Matmul::Metadata(lhs_t, res_v, rhs_v);
}

const char *NodeMatmul::node_type() const noexcept { return "NodeMatmul"; }

void NodeMatmul::eval() {
    if (!this->evaluated()) {
        this->lhs->eval();
        this->rhs->eval();

        forward_op(this->lhs->value().data(), this->rhs->value().data(),
                   this->value().data(), this->forward);

        this->evaluated_ = true;
    }
}

void NodeMatmul::get_grad() {
    backward_op(this->lhs->value().data(), this->lhs->gradient().data(),
                this->rhs->value().data(), this->rhs->gradient().data(),
                this->gradient().data(), this->backward_wrt_lhs,
                this->backward_wrt_rhs);

    if (!this->lhs->is_input()) {
        this->lhs->get_grad();
    }

    if (!this->rhs->is_input()) {
        this->rhs->get_grad();
    }
}

} // namespace kaad
