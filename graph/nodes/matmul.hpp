#pragma once

#include "../../tensorfuncs/adjoint_ops.hpp" // for tensorfuncs::adjoint
#include "../../tensorfuncs/primal_ops.hpp"  // for tensorfuncs::primal
#include "inode.hpp"                         // for INode

namespace kaad {

/**
 * @brief A matmul node in a computation graph.
 * @see tensorfuncs::primal::binary::matmul
 * @see tensorfuncs::adjoint::binary::matmul
 */
class Node_matmul : public INode {
  public:
    /**
     * @brief Returns the type of the node as a string.
     */
    const char *node_type() const noexcept override;

    INode *lhs = nullptr; ///< Pointer to the first input Node.
    INode *rhs = nullptr; ///< Pointer to the second input Node.

    tensorfuncs::primal::binary::matmul_fn<Scalar> forward_op =
        tensorfuncs::primal::binary::matmul; ///< Function pointer to the matmul
                                             ///< operation.
    tensorfuncs::adjoint::binary::matmul_fn<Scalar> backward_op =
        tensorfuncs::adjoint::binary::matmul; ///< Function pointer to the
                                              ///< matmul gradient.

    /**
     * @brief Stride arrays for tensors A, B, and C for all computation stages.
     * Each stride array is a flattened array of size 6, containing strides for
     * all 3 passes (forward and both backward gradients).
     * Index layout for each stride array:
     * - [0..1] Forward pass (C = A * B)
     * - [2..3] Gradient w.r.t. A (dA = dC * Bᵗ)
     * - [4..5] Gradient w.r.t. B (dB = Aᵗ * dC)
     */
    int lhs_rows[3]; ///< Number of rows of tensor A for each computation stage.
    int rhs_cols[3]; ///< Number of columns of tensor B for each computation
                     ///< stage.
    int shared_dim[3]; ///< Shared inner dimension for each computation stage.
    int lhs_stride[6]; ///< Flattened stride pairs for tensor A (2 per stage × 3
                       ///< stages).
    int rhs_stride[6]; ///< Flattened stride pairs for tensor B (2 per stage × 3
                       ///< stages).
    int value_stride[6]; ///< Flattened stride pairs for tensor C (2 per stage ×
                         ///< 3 stages).

    void metadata();

    /**
     * @brief Constructs a matmul node.
     * @param lhs_ptr Pointer to the first input node.
     * @param rhs_ptr Pointer to the second input node.
     * @param value_shape Shape of the value and gradient tensors.
     */
    Node_matmul(INode *lhs_ptr, INode *rhs_ptr,
                std::span<const int> value_shape);

    /**
     * @brief Evaluates the matmul operation by apllying forward_op,if not
     * already evaluated.
     */
    void eval() override;

    /**
     * @brief Propagates gradients back through the matmul operation, by
     * applying backward_op.
     */
    inline void getGrad() override;
};

} // namespace kaad
