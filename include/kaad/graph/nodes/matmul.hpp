#pragma once

#include "../../functions/adjoint.hpp" // for matmul, matmul_fn
#include "../../functions/primal.hpp"  // for matmul, matmul_fn
#include "../../scalar.hpp"            // for Scalar
#include "inode.hpp"                   // for INode
#include <span>                        // for span

namespace kaad {

/**
 * @brief A matmul node in a computation graph.
 * @ingroup nodes
 * @see functions::primal::binary::matmul
 * @see functions::adjoint::binary::matmul
 */
class Node_matmul : public INode {
  private:
    INode *lhs = nullptr; ///< Pointer to the first input Node.
    INode *rhs = nullptr; ///< Pointer to the second input Node.

    functions::primal::binary::matmul_fn<Scalar> forward_op =
        functions::primal::binary::matmul; ///< Function pointer to the matmul
                                           ///< operation.
    functions::adjoint::binary::matmul_fn<Scalar> backward_op =
        functions::adjoint::binary::matmul; ///< Function pointer to the
                                            ///< matmul gradient.

    /**
     * @brief Stride arrays for tensors A, B, and C for all computation stages.
     * @ingroup nodes
     * Each stride array is a flattened array of size 6, containing strides for
     * all 3 passes (forward and both backward gradients).
     * Index layout for each stride array:
     * - [0..1] Forward pass (C = A * B)
     * - [2..3] Gradient w.r.t. A (dA = dC * B^t)
     * - [4..5] Gradient w.r.t. B (dB = A^t * dC)
     */
    int lhs_rows[3]; ///< Number of rows of tensor A for each computation stage.
    int rhs_cols[3]; ///< Number of columns of tensor B for each computation
                     ///< stage.
    int shared_dim[3]; ///< Shared inner dimension for each computation stage.
    int lhs_stride[6]; ///< Flattened stride pairs for tensor A (2 per stage x 3
                       ///< stages).
    int rhs_stride[6]; ///< Flattened stride pairs for tensor B (2 per stage x 3
                       ///< stages).
    int value_stride[6]; ///< Flattened stride pairs for tensor C (2 per stage x
                         ///< 3 stages).

    void metadata();

  public:
    /**
     * @brief Constructs a matmul node.
     * @ingroup nodes
     * @param lhs_ptr Pointer to the first input node.
     * @param rhs_ptr Pointer to the second input node.
     * @param value_shape Shape of the value and gradient tensors.
     */
    Node_matmul(INode *lhs_ptr, INode *rhs_ptr,
                std::span<const int> value_shape);

    /**
     * @brief Returns the type of the node as a string.
     * @ingroup nodes
     */
    const char *node_type() const noexcept override;

    /**
     * @brief Evaluates the matmul operation by apllying forward_op,if not
     * @ingroup nodes
     * already evaluated.
     */
    void eval() override;

    /**
     * @brief Propagates gradients back through the matmul operation, by
     * @ingroup nodes
     * applying backward_op.
     */
    inline void getGrad() override;
};

} // namespace kaad
