#include "dot.hpp"

#include <array>                          // for array
#include <kaad/enums.hpp>                 // for ScalarOrder
#include <kaad/functions/dot_product.hpp> // for DotProduct
#include <kaad/graph/nodes/inode.hpp>     // for INode
#include <kaad/scalar.hpp>                // for Scalar
#include <kaad/tensor/tensor.hpp>         // for Tensor

namespace kaad {

NodeDot::NodeDot(INode *lhs_ptr, INode *rhs_ptr)
    : INode(std::array<int, 0>{}, false), lhs(lhs_ptr), rhs(rhs_ptr) {

    const Tensor &lhs_val = this->lhs->value();
    const Tensor &rhs_val = this->rhs->value();

    const Scalar *lhs_end = lhs_val.data() + lhs_val.size();
    const Scalar *rhs_end = rhs_val.data() + rhs_val.size();

    if (lhs_val.scalar()) {

        this->forward_op = functions::DotProduct::primal<LHS_IS_SCALAR>;
        this->backward_op = functions::DotProduct::adjoint<LHS_IS_SCALAR>;
        this->end = rhs_end;

    } else if (rhs_val.scalar()) {

        this->forward_op = functions::DotProduct::primal<RHS_IS_SCALAR>;
        this->backward_op = functions::DotProduct::adjoint<RHS_IS_SCALAR>;
        this->end = lhs_end;
    } else {
        // both inputs are rank-1
        this->end = lhs_end;
    }
}

const char *NodeDot::node_type() const noexcept { return "NodeDot"; }

void NodeDot::eval() {
    if (!this->evaluated()) {
        this->lhs->eval();
        this->rhs->eval();

        forward_op(this->lhs->value().data(), this->rhs->value().data(),
                   this->value().data(), end);
        this->evaluated_ = true;
    }
}

void NodeDot::get_grad() {
    backward_op(this->lhs->value().data(), this->lhs->gradient().data(),
                this->rhs->value().data(), this->rhs->gradient().data(),
                this->gradient().data(), end);

    if (!this->lhs->is_input()) {
        this->lhs->get_grad();
    }
    if (!this->rhs->is_input()) {
        this->rhs->get_grad();
    }
}

} // namespace kaad
