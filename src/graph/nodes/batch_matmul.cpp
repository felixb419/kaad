#include <kaad/graph/nodes/batch_matmul.hpp>

#include "kaad/functions/batch_matmul.hpp" // for BatchMatmul
#include "kaad/tensor/tensor.hpp"          // for Tensor
#include <kaad/graph/nodes/inode.hpp>      // for INode
#include <kaad/scalar.hpp>                 // for Scalar
#include <kaad/tensor/tensor_view.hpp>     // for TensorView

namespace kaad {

Node_batch_matmul::Node_batch_matmul(INode *lhs_ptr, INode *rhs_ptr,
                                     Tensor::Shape_view value_s)
    : INode(value_s, false), lhs(lhs_ptr), rhs(rhs_ptr) {

    TensorView lhs_v = this->lhs->value().view();
    TensorView rhs_v = this->rhs->value().view();
    TensorView res_v = this->value().view();

    // make lhs^T
    Tensor::Shape lhs_shape_buff;
    Tensor::Stride lhs_stride_buff;
    TensorView lhs_T = lhs_v.transpose_2d(lhs_shape_buff, lhs_stride_buff);

    // make rhs^T
    Tensor::Shape rhs_shape_buff;
    Tensor::Stride rhs_stride_buff;
    TensorView rhs_T = rhs_v.transpose_2d(rhs_shape_buff, rhs_stride_buff);

    // Compute metadata for individual passes
    // lhs * rhs = res
    this->forward = functions::BatchMatmul::Metadata(lhs_v, rhs_v, res_v);

    // d_res * rhs^T = d_lhs
    this->backward_wrt_lhs =
        functions::BatchMatmul::Metadata(res_v, rhs_T, lhs_v);

    // lhs^T * d_res = d_rhs
    this->backward_wrt_rhs =
        functions::BatchMatmul::Metadata(lhs_T, res_v, rhs_v);

    // assign compile-time recursive functions
    auto fn_pair = functions::BatchMatmul::dispatch(this->value().rank());
    this->forward_op = fn_pair.primal;
    this->backward_op = fn_pair.adjoint;
}

const char *Node_batch_matmul::node_type() const noexcept {
    return "Node_batch_matmul";
}

void Node_batch_matmul::eval() {

    if (!this->evaluated()) {
        this->lhs->eval();
        this->rhs->eval();

        const Scalar *lhs = this->lhs->value().data();
        const Scalar *rhs = this->rhs->value().data();
        Scalar *res = this->value().data();

        this->forward_op(lhs, rhs, res, this->forward);

        this->evaluated_ = true;
    }
}

void Node_batch_matmul::getGrad() {

    const Scalar *lhs = this->lhs->value().data();
    Scalar *d_lhs = this->lhs->gradient().data();
    const Scalar *rhs = this->rhs->value().data();
    Scalar *d_rhs = this->rhs->gradient().data();
    const Scalar *d_res = this->gradient().data();

    this->backward_op(lhs, d_lhs, rhs, d_rhs, d_res, this->backward_wrt_lhs,
                      this->backward_wrt_rhs);

    if (!this->lhs->isInput()) {
        this->lhs->getGrad();
    }
    if (!this->rhs->isInput()) {
        this->rhs->getGrad();
    }
}

} // namespace kaad
