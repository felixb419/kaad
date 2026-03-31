#include "matmul.hpp"

#include <kaad/functions/matmul.hpp>    // for Matmul
#include <kaad/graph/nodes/inode.hpp>   // for INode
#include <kaad/tensor/tensor.hpp>       // for Tensor
#include <kaad/tensor/tensor_types.hpp> // for Shape, Strides, ShapeView
#include <kaad/tensor/tensor_view.hpp>  // for TensorView

namespace kaad {

NodeMatmul::NodeMatmul(INode *lhs_ptr, INode *rhs_ptr, ShapeView value_shape)
    : INode(value_shape, false), lhs(lhs_ptr), rhs(rhs_ptr) {

    TensorView lhs_v = this->lhs->value().view();
    TensorView rhs_v = this->rhs->value().view();
    TensorView res_v = this->value().view();

    // make lhs^T
    Shape lhs_shape_buff;
    Strides lhs_strides_buff;
    TensorView lhs_t = lhs_v.transpose_2d(lhs_shape_buff, lhs_strides_buff);

    // make rhs^T
    Shape rhs_shape_buff;
    Strides rhs_strides_buff;
    TensorView rhs_t = rhs_v.transpose_2d(rhs_shape_buff, rhs_strides_buff);

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
