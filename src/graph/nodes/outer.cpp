#include <kaad/graph/nodes/outer.hpp>

#include <algorithm>                  // for copy
#include <array>                      // for array
#include <kaad/graph/dispatchers.hpp> // for get_flexGrad
#include <kaad/graph/nodes/inode.hpp> // for INode
#include <kaad/tensor/tensor.hpp>     // for Tensor

namespace kaad {

void Node_outer::metadata() {
    // compute metadata
    Tensor &lhs = this->lhs->value();
    Tensor &rhs = this->rhs->value();
    Tensor &res = this->value();

    this->res_rank = res.rank();

    this->lhs_stride.resize(this->res_rank);
    this->rhs_stride.resize(this->res_rank);
    this->res_stride.resize(this->res_rank);

    std::copy(res.stride().begin(), res.stride().end(),
              this->res_stride.data());
    std::copy(lhs.stride().begin(), lhs.stride().end(),
              this->lhs_stride.data());
    std::copy(rhs.stride().begin(), rhs.stride().end(),
              this->rhs_stride.data() + lhs.rank());

    this->res_offset.resize(this->res_rank);
    for (std::size_t i = 0; i < this->res_rank; i++) {
        this->res_offset[i] =
            static_cast<std::size_t>(res.shape()[i]) * this->res_stride[i];
    }

    // assign compile-time recursive function
    if (this->res_rank <= Dispatchers::MAX_NDIMS) {
        this->forward_op =
            Dispatchers::get_flexOp<Node_outer::Kernel>()[this->res_rank];
        this->backward_op =
            Dispatchers::get_flexGrad<Node_outer::Kernel>()[this->res_rank];
    }
}

Node_outer::Node_outer(INode *lhs_ptr, INode *rhs_ptr,
                       std::span<const int> value_shape)
    : INode(value_shape, false), lhs(lhs_ptr), rhs(rhs_ptr) {

    this->metadata();
}

const char *Node_outer::node_type() const noexcept { return "Node_outer"; }

void Node_outer::eval() {
    if (!this->evaluated()) {
        this->lhs->eval();
        this->rhs->eval();

        forward_op(this->lhs->value().data(), this->rhs->value().data(),
                   this->value().data(), lhs_stride.data(), rhs_stride.data(),
                   res_stride.data(), res_offset.data(), res_rank);
        this->evaluated_ = true;
    }
}

void Node_outer::getGrad() {
    backward_op(this->lhs->value().data(), this->lhs->gradient().data(),
                this->rhs->value().data(), this->rhs->gradient().data(),
                this->value().data(), this->gradient().data(),
                lhs_stride.data(), rhs_stride.data(), res_stride.data(),
                res_offset.data(), res_rank);

    if (!this->lhs->isInput()) {
        this->lhs->getGrad();
    }
    if (!this->rhs->isInput()) {
        this->rhs->getGrad();
    }
}

} // namespace kaad
