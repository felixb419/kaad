#pragma once

#include <kaad/functions/batch_matmul.hpp> // for BatchMatmul
#include <kaad/graph/nodes/inode.hpp>      // for INode
#include <span>                            // for span

namespace kaad {

/**
 * @brief Batch matrix multiplication node.
 * @internal
 * @ingroup nodes
 * @see functions::primal::binary::batch_matmul
 * @see functions::adjoint::binary::batch_matmul
 */
class Node_batch_matmul : public INode {
  private:
    INode *lhs = nullptr;
    INode *rhs = nullptr;

    functions::BatchMatmul::primal_fn forward_op;
    functions::BatchMatmul::adjoint_fn backward_op;

    functions::BatchMatmul::Metadata forward;
    functions::BatchMatmul::Metadata backward_wrt_lhs;
    functions::BatchMatmul::Metadata backward_wrt_rhs;

  public:
    [[nodiscard]] const char *node_type() const noexcept override;

    /**
     * @brief Construct batch_matmul node.
     * @param lhs_ptr Pointer to the first input node.
     * @param rhs_ptr Pointer to the second input node.
     * @param value_shape Output/gradient shape
     */
    Node_batch_matmul(INode *lhs_ptr, INode *rhs_ptr,
                      std::span<const int> value_shape);

    ~Node_batch_matmul() noexcept override = default;

    /// Compute elements for @c value .
    /// Computes values for @c lhs and @c rhs first.
    void eval() override;

    /// Compute elements for @c lhs::gradient and @c rhs::gradient.
    /// Computes gradients for @c lhs and @c rhs after.
    void getGrad() override;
};

} // namespace kaad
