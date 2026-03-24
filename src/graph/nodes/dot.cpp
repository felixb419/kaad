#include <kaad/graph/nodes/dot.hpp>

#include <array>                      // for array
#include <kaad/graph/nodes/inode.hpp> // for INode
#include <kaad/tensor/tensor.hpp>     // for Tensor
#include <span>                       // for span

namespace kaad {

NodeDot::NodeDot(INode *lhs_ptr, INode *rhs_ptr)
    : INode(std::array<int, 0>{}, false), lhs(lhs_ptr), rhs(rhs_ptr) {

    this->lhs_end = this->lhs->value().data() + this->lhs->value().size();
}

const char *NodeDot::node_type() const noexcept { return "NodeDot"; }

void NodeDot::eval() {
    if (!this->evaluated()) {
        this->lhs->eval();
        this->rhs->eval();

        forward_op(this->lhs->value().data(), this->rhs->value().data(),
                   this->value().data(), lhs_end);
        this->evaluated_ = true;
    }
}

void NodeDot::get_grad() {
    backward_op(this->lhs->value().data(), this->lhs->gradient().data(),
                this->rhs->value().data(), this->rhs->gradient().data(),
                this->gradient().data(), lhs_end);

    if (!this->lhs->is_input()) {
        this->lhs->get_grad();
    }
    if (!this->rhs->is_input()) {
        this->rhs->get_grad();
    }
}

} // namespace kaad
