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
  private:
    void metadata(const int *offset_arr);

  public:
    const char *node_type() const noexcept override;

    tensorfuncs::primal::unary::slice_fn<Scalar> forward_op =
        tensorfuncs::primal::unary::slice;
    tensorfuncs::adjoint::unary::slice_fn<Scalar> backward_op =
        tensorfuncs::adjoint::unary::slice;

    std::vector<int> strideA;           ///< Stride array for tensor A.
    std::vector<int> strideC;           ///< Stride array for tensor C.
    std::vector<size_t> start_offset_a; ///< Offset for the start of A.
    std::vector<size_t> C_offset; ///< Per-dim offset to the end of C buffer.
    size_t C_nDims = 0;           ///< Number of the dimensions of the C tensor.

    /**
     * @brief Constructs a slice node.
     * @param A_ptr    Pointer to the input node.
     * @param tensor_args       Arguments to construct the output tensor.
     */
    template <typename... TensorArgs>
    Node_slice(INode *A_ptr, const int *offset_arr, TensorArgs &&...tensor_args)
        : INode(A_ptr, tensor_args...) {
        this->metadata(offset_arr);
    }

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
