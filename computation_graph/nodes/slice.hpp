#pragma once

#include "../../tensorfuncs/adjoint_ops.hpp" // for tensorfuncs::adjoint
#include "../../tensorfuncs/primal_ops.hpp"  // for tensorfuncs::primal
#include "inode.hpp"                         // for INode

namespace kaad {

/**
 * @brief A slice operation node in a computation graph.
 * @see tensorfuncs::primal::unary::slice
 * @see tensorfuncs::adjoint::unary::slice
 */
class Node_slice : public INode {
  public:
    /**
     * @brief Returns the type of the node as a string.
     */
    const char *node_type() const noexcept override;

    INode *input = nullptr; ///< Pointer to the input Node.

    tensorfuncs::primal::unary::slice_fn<Scalar> forward_op =
        tensorfuncs::primal::unary::slice;
    tensorfuncs::adjoint::unary::slice_fn<Scalar> backward_op =
        tensorfuncs::adjoint::unary::slice;

    std::vector<int> input_stride;      ///< Stride array for tensor input.
    std::vector<int> value_stride;      ///< Stride array for tensor value.
    std::vector<size_t> start_offset_a; ///< Offset for the start of input.
    std::vector<size_t>
        value_offset;      ///< Per-dim offset to the end of value buffer.
    size_t value_rank = 0; ///< Number of the dimensions of the value tensor.

    /**
     * @brief Constructs a slice node.
     * @param input_ptr    Pointer to the input node.
     * @param offset_arr Array with the per-dim offsets of the slice.
     * @param value_shape Shape of the value and gradient tensors.
     */
    Node_slice(INode *input_ptr, const int *offset_arr,
               std::span<const int> value_shape);

    /**
     * @brief Evaluates the slice operation by applying forward_op, if not
     * already evaluated.
     */
    void eval() override;

    /**
     * @brief Propagates gradients back through the slice operation by applying
     * backward_op.
     */
    void getGrad() override;
};

} // namespace kaad
