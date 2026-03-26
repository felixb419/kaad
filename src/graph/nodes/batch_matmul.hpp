#pragma once

#include <kaad/functions/batch_matmul.hpp> // for BatchMatmul
#include <kaad/graph/nodes/inode.hpp>      // for INode
#include <kaad/tensor/tensor_types.hpp>    // for ShapeView

namespace kaad {

/**
 * @brief A batch matmuloperation node for a @ref kaad::Graph
 * @ingroup nodes
 * @internal
 */
class NodeBatchMatmul : public INode {
  private:
    INode *lhs = nullptr;
    INode *rhs = nullptr;

    functions::BatchMatmul::primal_fn forward_op;
    functions::BatchMatmul::adjoint_fn backward_op;

    functions::BatchMatmul::Metadata forward;
    functions::BatchMatmul::Metadata backward_wrt_lhs;
    functions::BatchMatmul::Metadata backward_wrt_rhs;

  public:
    /// @return Type of the node as a string.
    [[nodiscard]] const char *node_type() const noexcept override;

    /**
     * @brief Construct batch_matmul node.
     * @param lhs_ptr Pointer to the first input node.
     * @param rhs_ptr Pointer to the second input node.
     * @param value_shape Output/gradient shape
     */
    NodeBatchMatmul(INode *lhs_ptr, INode *rhs_ptr, ShapeView value_shape);

    ~NodeBatchMatmul() noexcept override = default;

    /// Compute @c value for this node.
    /// Computes @c value for @c lhs and @c rhs first.
    void eval() override;

    /// Compute @c gradient for this node.
    /// Computes @c gradient for @c lhs and @c rhs after.
    void get_grad() override;
};

} // namespace kaad
