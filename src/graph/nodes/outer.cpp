#include "outer.hpp"

#include <algorithm>                    // for copy
#include <array>                        // for array
#include <kaad/graph/dispatchers.hpp>   // for get_flexGrad, get_flexOp
#include <kaad/graph/nodes/inode.hpp>   // for INode
#include <kaad/max_rank.hpp>            // for KAAD_MAX_RANK
#include <kaad/tensor/tensor.hpp>       // for Tensor
#include <kaad/tensor/tensor_types.hpp> // for ShapeView

namespace kaad {

void NodeOuter::metadata() {
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
    if (this->res_rank <= KAAD_MAX_RANK) {
        this->forward_op =
            Dispatchers::get_flexOp<NodeOuter::Kernel>()[this->res_rank];
        this->backward_op =
            Dispatchers::get_flexGrad<NodeOuter::Kernel>()[this->res_rank];
    }
}

NodeOuter::NodeOuter(INode *lhs_ptr, INode *rhs_ptr, ShapeView value_shape)
    : INode(value_shape, false), lhs(lhs_ptr), rhs(rhs_ptr) {

    this->metadata();
}

const char *NodeOuter::node_type() const noexcept { return "NodeOuter"; }

void NodeOuter::eval() {
    if (!this->evaluated()) {
        this->lhs->eval();
        this->rhs->eval();

        forward_op(this->lhs->value().data(), this->rhs->value().data(),
                   this->value().data(), lhs_stride.data(), rhs_stride.data(),
                   res_stride.data(), res_offset.data(), res_rank);
        this->evaluated_ = true;
    }
}

void NodeOuter::get_grad() {
    backward_op(this->lhs->value().data(), this->lhs->gradient().data(),
                this->rhs->value().data(), this->rhs->gradient().data(),
                this->value().data(), this->gradient().data(),
                lhs_stride.data(), rhs_stride.data(), res_stride.data(),
                res_offset.data(), res_rank);

    if (!this->lhs->is_input()) {
        this->lhs->get_grad();
    }
    if (!this->rhs->is_input()) {
        this->rhs->get_grad();
    }
}

} // namespace kaad
