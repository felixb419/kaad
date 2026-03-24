#include <kaad/graph/nodes/mean_dim.hpp>

#include <array>                      // for array
#include <kaad/graph/common.hpp>      // for along_dim_met...
#include <kaad/graph/dispatchers.hpp> // for get_meanDim
#include <kaad/graph/nodes/inode.hpp> // for INode
#include <kaad/max_rank.hpp>          // for KAAD_MAX_RANK
#include <kaad/scalar.hpp>            // for Scalar
#include <kaad/tensor/tensor.hpp>     // for Tensor

namespace kaad {

void NodeMeanDim::metadata(int dim) {

    // compute metadata
    Tensor &input = this->input->value();
    Tensor &value = this->value();
    Tensor &input_grad = this->input->gradient();

    this->divisor = static_cast<Scalar>(input.shape()[dim]);
    this->value_end = value.data() + value.size();
    this->input_grad_end = input_grad.data() + input_grad.size();

    detail::along_dim_metadata_impl(input, value, dim, this->input_rank,
                                    this->input_offset, this->input_stride,
                                    this->value_stride);

    // assign compile-time recursive function
    std::size_t input_rank = input.rank();
    if (input_rank <= KAAD_MAX_RANK) {
        this->forward_op = Dispatchers::get_meanDim<Scalar>()[input_rank];
        this->backward_op = Dispatchers::get_meanDim_grad<Scalar>()[input_rank];
    }
}

NodeMeanDim::NodeMeanDim(INode *input_ptr, int dim,
                         std::span<const int> value_shape)
    : INode(value_shape, false), input(input_ptr) {

    this->metadata(dim);
}

const char *NodeMeanDim::node_type() const noexcept { return "NodeMeanDim"; }

void NodeMeanDim::eval() {
    if (!this->evaluated()) {
        this->input->eval();

        forward_op(this->input->value().data(), this->value().data(),
                   input_stride.data(), value_stride.data(),
                   input_offset.data(), input_rank, divisor, value_end);
        this->evaluated_ = true;
    }
}

void NodeMeanDim::getGrad() {
    backward_op(this->input->gradient().data(), this->gradient().data(),
                input_stride.data(), value_stride.data(), input_offset.data(),
                input_rank, divisor, input_grad_end);

    if (!this->input->isInput()) {
        this->input->getGrad();
    }
}

} // namespace kaad
