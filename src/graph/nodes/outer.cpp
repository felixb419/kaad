#include "outer.hpp"

#include <algorithm>                    // for copy
#include <cstddef>                      // for size_t
#include <kaad/functions/flexible.hpp>  // for Flexible
#include <kaad/functions/kernels.hpp>   // for Mul
#include <kaad/graph/nodes/inode.hpp>   // for INode
#include <kaad/static_vector.hpp>       // for StaticVector
#include <kaad/tensor/tensor.hpp>       // for Tensor
#include <kaad/tensor/tensor_types.hpp> // for ShapeView

namespace kaad {

void NodeOuter::metadata() {
    // compute metadata
    Tensor &lhs = this->lhs->value();
    Tensor &rhs = this->rhs->value();
    Tensor &res = this->value();

    this->mdata.eff_lhs.resize(res.rank());
    this->mdata.eff_rhs.resize(res.rank());
    this->mdata.eff_res.resize(res.rank());

    std::copy(res.strides().begin(), res.strides().end(),
              this->mdata.eff_res.data());
    std::copy(lhs.strides().begin(), lhs.strides().end(),
              this->mdata.eff_lhs.data());
    std::copy(rhs.strides().begin(), rhs.strides().end(),
              this->mdata.eff_rhs.data() + lhs.rank());

    this->mdata.res_ends.resize(res.rank());
    for (std::size_t i = 0; i < res.rank(); i++) {
        this->mdata.res_ends[i] =
            static_cast<std::size_t>(res.shape()[i]) * this->mdata.eff_res[i];
    }

    // assign compile-time recursive function
    auto fn_pair = functions::Flexible::dispatch<Kernel>(res.rank());
    this->forward_op = fn_pair.primal;
    this->backward_op = fn_pair.adjoint;
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
                   this->value().data(), this->mdata);
        this->evaluated_ = true;
    }
}

void NodeOuter::get_grad() {
    backward_op(this->lhs->value().data(), this->lhs->gradient().data(),
                this->rhs->value().data(), this->rhs->gradient().data(),
                this->value().data(), this->gradient().data(), this->mdata);

    if (!this->lhs->is_input()) {
        this->lhs->get_grad();
    }
    if (!this->rhs->is_input()) {
        this->rhs->get_grad();
    }
}

} // namespace kaad
