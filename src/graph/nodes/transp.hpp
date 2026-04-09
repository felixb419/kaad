#pragma once

#include <kaad/functions/kernels.hpp>   // for NoOp
#include <kaad/functions/pointwise.hpp> // for Pointwise
#include <kaad/graph/nodes/inode.hpp>   // for INode
#include <kaad/scalar.hpp>              // for Scalar
#include <kaad/tensor/tensor_types.hpp> // for ShapeView, StridesView

namespace kaad {

/**
 * @brief A transposition operation node for a @ref kaad::Graph
 * @ingroup nodes
 * @internal
 */
class NodeTransp : public INode {
  private:
    INode *input = nullptr; ///< Pointer to the input Node.

    using Kernel = Kernels::NoOp<Scalar>;

    functions::Pointwise::Unary::primal_fn<Kernel> forward_op =
        functions::Pointwise::Unary::primal<Kernel>;

    functions::Pointwise::Unary::adjoint_fn<Kernel> backward_op =
        functions::Pointwise::Unary::adjoint<Kernel>;

    const Scalar *input_end =
        nullptr; ///< Pointer to the end of the input buffer.
    const Scalar *value_end =
        nullptr; ///< Pointer to the end of the value buffer.

  public:
    /**
     * @brief Construct tranposition node.
     * @param input_ptr Pointer to the first input node.
     * @param value_shape Output/gradient shape
     * @param value_shape Output/gradient strides
     */
    NodeTransp(INode *input_ptr, ShapeView value_shape,
               StridesView value_strides);

    /// @return Type of the node as a string.
    [[nodiscard]] const char *node_type() const noexcept override;

    /// Compute @c value for this node.
    /// Computes @c value for @c lhs and @c rhs first.
    void eval() override;

    /// Compute @c gradient for this node.
    /// Computes @c gradient for @c lhs and @c rhs after.
    void get_grad() override;
};

} // namespace kaad
