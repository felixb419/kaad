#pragma once

#include <kaad/functions/adjoint.hpp>   // for pointwise, pointwise_fn
#include <kaad/functions/kernels.hpp>   // for NoOp
#include <kaad/functions/primal.hpp>    // for pointwise, pointwise_fn
#include <kaad/graph/nodes/inode.hpp>   // for INode
#include <kaad/scalar.hpp>              // for Scalar
#include <kaad/tensor/tensor_types.hpp> // for ShapeView, StrideView

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

    functions::primal::unary::pointwise_fn<Kernel> forward_op =
        functions::primal::unary::pointwise<Kernel>; ///< Function pointer to
                                                     ///< the value operation.

    functions::adjoint::unary::pointwise_fn<Kernel> backward_op =
        functions::adjoint::unary::pointwise<Kernel>; ///< Function pointer to
                                                      ///< the gradient
                                                      ///< operation.

    const Scalar *input_end =
        nullptr; ///< Pointer to the end of the input buffer.
    const Scalar *value_end =
        nullptr; ///< Pointer to the end of the value buffer.

  public:
    /**
     * @brief Construct tranposition node.
     * @param input_ptr Pointer to the first input node.
     * @param value_shape Output/gradient shape
     * @param value_shape Output/gradient stride
     */
    NodeTransp(INode *input_ptr, ShapeView value_shape,
               StrideView value_stride);

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
