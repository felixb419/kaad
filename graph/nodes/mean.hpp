#pragma once

#include "../../functions/adjoint_ops.hpp" // for mean, mean_fn
#include "../../functions/primal_ops.hpp"  // for mean, mean_fn
#include "../../scalar.hpp"                // for Scalar
#include "inode.hpp"                       // for INode
#include <span>                            // for span

namespace kaad {

/**
 * @brief A mean operation node in a computation graph.
 * @see functions::primal::unary::mean
 * @see functions::adjoint::unary::mean
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
     * @brief Constructs a mean node.
     * @param input_ptr Pointer to the input node.
     * @param value_shape Shape of the value and gradient tensors.
     */
    Node_mean(INode *input_ptr, std::span<const int> value_shape);

    /**
     * @brief Returns the type of the node as a string.
     */
    const char *node_type() const noexcept override;

    /**
     * @brief Evaluates the mean operation by applying forwrd_op, if not already
     * evaluated.
     */
    void eval() override;

    /**
     * @brief Propagates gradients back through the mean operation, by applying
     * backward_op.
     */
    void getGrad() override;
};

} // namespace kaad
