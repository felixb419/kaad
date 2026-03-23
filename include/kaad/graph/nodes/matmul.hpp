#pragma once

#include <array>                      // for array
#include <kaad/functions/adjoint.hpp> // for matmul, matmul_fn
#include <kaad/functions/primal.hpp>  // for matmul, matmul_fn
#include <kaad/graph/nodes/inode.hpp> // for INode
#include <kaad/scalar.hpp>            // for Scalar
#include <span>                       // for span

namespace kaad {

/**
 * @brief A matmul operation node for a @ref kaad::Graph
 * @ingroup nodes
 * @internal
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
    std::array<int, 3>
        lhs_rows; ///< Number of rows of tensor A for each computation stage.
    std::array<int, 3> rhs_cols; ///< Number of columns of tensor B for each
                                 ///< computation stage.
    std::array<int, 3>
        shared_dim; ///< Shared inner dimension for each computation stage.

    // NOLINTBEGIN(readability-magic-numbers)
    std::array<int, 6> lhs_stride;   ///< Flattened stride pairs for tensor A (2
                                     ///< per stage x 3 stages).
    std::array<int, 6> rhs_stride;   ///< Flattened stride pairs for tensor B (2
                                     ///< per stage x 3 stages).
    std::array<int, 6> value_stride; ///< Flattened stride pairs for tensor C (2
                                     ///< per stage x 3 stages).
    // NOLINTEND(readability-magic-numbers)

    void metadata();

  public:
    /**
     * @brief Construct matmul node.
     * @param lhs_ptr Pointer to the first input node.
     * @param rhs_ptr Pointer to the second input node.
     * @param value_shape Output/gradient shape
     */
    Node_matmul(INode *lhs_ptr, INode *rhs_ptr,
                std::span<const int> value_shape);

    /// @return Type of the node as a string.
    [[nodiscard]] const char *node_type() const noexcept override;

    /// Compute @c value for this node.
    /// Computes @c value for @c lhs and @c rhs first.
    void eval() override;

    /// Compute @c gradient for this node.
    /// Computes @c gradient for @c lhs and @c rhs after.
    void getGrad() override;
};

} // namespace kaad
