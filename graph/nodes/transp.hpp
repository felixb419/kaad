#pragma once

#include "../../functions/adjoint_ops.hpp" // for functions::adjoint
#include "../../functions/kernels.hpp"     // for NoOp
#include "../../functions/primal_ops.hpp"  // for functions::primal
#include "inode.hpp"                         // for INode

namespace kaad {

/**
 * @brief A transpose operation node in a computation graph.
 * @see functions::primal::unary::noop
 * @see functions::adjoint::unary::pointwise
 */
class Node_transp : public INode {
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
     * @brief Constructs a transpose node with the given operation and gradient.
     *
     * @param input_ptr Pointer to the input node.
     * @param value_shape Shape of the value and gradient tensors.
     * @param value_stride Stride array of the value and gradient tensors.
     */
    Node_transp(INode *input_ptr, std::span<const int> value_shape,
                std::span<const int> value_stride);

    /**
     * @brief Returns the type of the node as a string.
     */
    const char *node_type() const noexcept override;

    /**
     * @brief Evaluates the transpose operation by applying forward_op, if not
     * already evaluated.
     */
    void eval() override;

    /**
     * @brief Propagates gradients back through the transpose operation, by
     * applying backward_op.
     */
    void getGrad() override;
};

} // namespace kaad
