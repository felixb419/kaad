#include "../../../include/kaad/graph/nodes/mean.hpp"
#include "../../../include/kaad/graph/nodes/inode.hpp" // for INode
#include "../../../include/kaad/scalar.hpp"            // for Scalar
#include "../../../include/kaad/tensor/tensor.hpp"     // for Tensor
#include <array>                                       // for array
#include <span>                                        // for span

namespace kaad {

Node_mean::Node_mean(INode *input_ptr)
    : INode(std::array<int, 0>{}, false), input(input_ptr) {
    this->input_end = input_ptr->value().data() + input_ptr->value().size();
    this->input_grad_end =
        input_ptr->gradient().data() + input_ptr->gradient().size();
    this->divisor = static_cast<Scalar>(input_ptr->value().size());
}

const char *Node_mean::node_type() const noexcept { return "Node_mean"; }

void Node_mean::eval() {
    if (!this->evaluated()) {
        this->input->eval();

        forward_op(this->input->value().data(), this->value().data(), input_end,
                   divisor);
        this->evaluated_ = true;
    }
}

void Node_mean::getGrad() {
    backward_op(this->input->gradient().data(), this->gradient().data(),
                input_grad_end, divisor);

    if (!this->input->isInput()) {
        this->input->getGrad();
    }
}

} // namespace kaad
