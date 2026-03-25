#include "transp.hpp"

#include <kaad/graph/nodes/inode.hpp>   // for INode
#include <kaad/tensor/tensor.hpp>       // for Tensor
#include <kaad/tensor/tensor_types.hpp> // for ShapeView, Stride_view

namespace kaad {

NodeTransp::NodeTransp(INode *input_ptr, ShapeView value_shape,
                       Stride_view value_stride)
    : INode(value_shape, false, "", value_stride), input(input_ptr) {
    // this->input_end = this->input->value().data() +
    // this->input->value().size();
    this->value_end = this->value().data() + this->value().size();
}

const char *NodeTransp::node_type() const noexcept { return "NodeTransp"; }

void NodeTransp::eval() {
    if (!this->evaluated()) {
        this->input->eval();

        forward_op(this->input->value().data(), this->value().data(),
                   value_end);
        this->evaluated_ = true;
    }
}

void NodeTransp::get_grad() {
    backward_op(this->input->value().data(), this->input->gradient().data(),
                this->value().data(), this->gradient().data(), value_end);

    if (!this->input->is_input()) {
        this->input->get_grad();
    }
}

} // namespace kaad
