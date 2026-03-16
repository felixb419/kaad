#include <kaad/graph/nodes/dot.hpp>

#include <array>                      // for array
#include <kaad/graph/nodes/inode.hpp> // for INode
#include <kaad/tensor/tensor.hpp>     // for Tensor
#include <span>                       // for span

namespace kaad {

Node_dot::Node_dot(INode *lhs_ptr, INode *rhs_ptr)
    : INode(std::array<int, 0>{}, false), lhs(lhs_ptr), rhs(rhs_ptr) {

    this->lhs_end = this->lhs->value().data() + this->lhs->value().size();
}

const char *Node_dot::node_type() const noexcept { return "Node_dot"; }

void Node_dot::eval() {
    if (!this->evaluated()) {
        this->lhs->eval();
        this->rhs->eval();

        forward_op(this->lhs->value().data(), this->rhs->value().data(),
                   this->value().data(), lhs_end);
        this->evaluated_ = true;
    }
}

void Node_dot::getGrad() {
    backward_op(this->lhs->value().data(), this->lhs->gradient().data(),
                this->rhs->value().data(), this->rhs->gradient().data(),
                this->gradient().data(), lhs_end);

    if (!this->lhs->isInput()) {
        this->lhs->getGrad();
    }
    if (!this->rhs->isInput()) {
        this->rhs->getGrad();
    }
}

} // namespace kaad
