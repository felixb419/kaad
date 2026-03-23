#pragma once

#include <kaad/functions/adjoint.hpp> // for mean, mean_fn
#include <kaad/functions/primal.hpp>  // for mean, mean_fn
#include <kaad/graph/nodes/inode.hpp> // for INode
#include <kaad/scalar.hpp>            // for Scalar

namespace kaad {

/**
 * @brief A mean operation node for a @ref kaad::Graph
 * @ingroup nodes
 * @internal
 */
class Node_mean : public INode {
  private:
    INode *input = nullptr; ///< Pointer to the input Node.

    functions::primal::unary::mean_fn<Scalar> forward_op =
        functions::primal::unary::mean;
    functions::adjoint::unary::mean_fn<Scalar> backward_op =
        functions::adjoint::unary::mean;

    const Scalar *input_end = nullptr; ///< Pointer to the end of the input
                                       ///< buffer (used for iteration).
    const Scalar *input_grad_end =
        nullptr; ///< Pointer to the end of the input_grad buffer (used for
                 ///< iteration).
    Scalar divisor = 0; ///< Divisor to compute the mean of the input tensor
                        ///< (length of input buffer).

  public:
    /**
     * @brief Construct mean node.
     * @param input_ptr Pointer to the input node.
     */
    Node_mean(INode *input_ptr);

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
