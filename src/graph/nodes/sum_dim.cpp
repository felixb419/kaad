#include <kaad/graph/nodes/sum_dim.hpp>

#include <array>                      // for array
#include <kaad/graph/common.hpp>      // for along_dim_met...
#include <kaad/graph/dispatchers.hpp> // for get_sumDim
#include <kaad/graph/nodes/inode.hpp> // for INode
#include <kaad/max_rank.hpp>          // for KAAD_MAX_RANK
#include <kaad/scalar.hpp>            // for Scalar
#include <kaad/tensor/tensor.hpp>     // for Tensor

namespace kaad {

void NodeSumDim::metadata(int dim) {
    // compute metadata
    Tensor &input = this->input->value();
    Tensor &value = this->value();

    detail::along_dim_metadata_impl(input, value, dim, this->value_rank,
                                    this->input_offset, this->input_stride,
                                    this->value_stride);

    // assign compile-time recursive function
    std::size_t input_rank = input.rank();
    if (input_rank <= KAAD_MAX_RANK) {
        this->val_func = Dispatchers::get_sumDim<Scalar>()[input_rank];
        this->grad_func = Dispatchers::get_sumDim_grad<Scalar>()[input_rank];
    }
}

NodeSumDim::NodeSumDim(INode *input_ptr, int dim,
                       std::span<const int> value_shape)
    : INode(value_shape, false), input(input_ptr) {

    this->metadata(dim);
}

const char *NodeSumDim::node_type() const noexcept { return "NodeSumDim"; }

void NodeSumDim::eval() {
    if (!this->evaluated()) {
        this->input->eval();

        val_func(this->input->value().data(), this->value().data(),
                 input_stride.data(), value_stride.data(), input_offset.data(),
                 value_rank);
        this->evaluated_ = true;
    }
}

void NodeSumDim::get_grad() {
    grad_func(this->input->gradient().data(), this->gradient().data(),
              input_stride.data(), value_stride.data(), input_offset.data(),
              value_rank);

    if (!this->input->is_input()) {
        this->input->get_grad();
    }
}

} // namespace kaad
