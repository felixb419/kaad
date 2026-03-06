#include "../../../include/kaad/graph/nodes/transp.hpp"
#include "../../../include/kaad/tensor/tensor.hpp" // for Tensor
#include <vector>                                  // for vector

namespace kaad {

Node_transp::Node_transp(INode *input_ptr, std::span<const int> value_shape,
                         std::span<const int> value_stride)
    : INode(value_shape, false, value_stride), input(input_ptr) {
    // this->input_end = this->input->value().data() +
    // this->input->value().size();
    this->value_end = this->value().data() + this->value().size();
}

const char *Node_transp::node_type() const noexcept { return "Node_transp"; }

void Node_transp::eval() {
    if (!this->evaluated()) {
        this->input->eval();

        forward_op(this->input->value().data(), this->value().data(),
                   value_end);
        this->evaluated_ = true;
    }
}

void Node_transp::getGrad() {
    backward_op(this->input->value().data(), this->input->gradient().data(),
                this->value().data(), this->gradient().data(), value_end);

    if (!this->input->isInput()) {
        this->input->getGrad();
    }
}

} // namespace kaad
