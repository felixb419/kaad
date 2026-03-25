#include "mean.hpp"

#include <array>                      // for array
#include <kaad/graph/nodes/inode.hpp> // for INode
#include <kaad/scalar.hpp>            // for Scalar
#include <kaad/tensor/tensor.hpp>     // for Tensor

namespace kaad {

NodeMean::NodeMean(INode *input_ptr)
    : INode(std::array<int, 0>{}, false), input(input_ptr) {
    this->input_end = input_ptr->value().data() + input_ptr->value().size();
    this->input_grad_end =
        input_ptr->gradient().data() + input_ptr->gradient().size();
    this->divisor = static_cast<Scalar>(input_ptr->value().size());
}

const char *NodeMean::node_type() const noexcept { return "NodeMean"; }

void NodeMean::eval() {
    if (!this->evaluated()) {
        this->input->eval();

        forward_op(this->input->value().data(), this->value().data(), input_end,
                   divisor);
        this->evaluated_ = true;
    }
}

void NodeMean::get_grad() {
    backward_op(this->input->gradient().data(), this->gradient().data(),
                input_grad_end, divisor);

    if (!this->input->is_input()) {
        this->input->get_grad();
    }
}

} // namespace kaad
