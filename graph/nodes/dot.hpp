#pragma once

#include "../../tensorfuncs/adjoint_ops.hpp" // for tensorfuncs::adjoint
#include "../../tensorfuncs/primal_ops.hpp"  // for tensorfuncs::primal
#include "inode.hpp"                         // for INode

namespace kaad {

/**
 * @brief A binary operation node in a computation graph.
 * @see tensorfuncs::primal::binary::pointwise
 * @see tensorfuncs::adjoint::binary::pointwise
 */
class Node_dot : public INode {
  public:
    /**
     * @brief Returns the type of the node as a string.
     */
    const char *node_type() const noexcept override;

    INode *lhs = nullptr; ///< Pointer to the first input Node.
    INode *rhs = nullptr; ///< Pointer to the second input Node.

    tensorfuncs::primal::binary::dot_fn<Scalar> forward_op =
        tensorfuncs::primal::binary::dot<Scalar>; ///< Function pointer to
                                                  ///< the value
                                                  ///< operation.

    tensorfuncs::adjoint::binary::dot_fn<Scalar> backward_op =
        tensorfuncs::adjoint::binary::dot<Scalar>; ///< Function pointer to the
                                                   ///< gradient operation.

    const Scalar *lhs_end =
        nullptr; ///< Pointer to the end of the A buffer (used for iteration).

    /**
     * @brief Constructs a binary operation node with the given operation and
     * gradient.
     * @param operation Function pointer to the value operation.
     * @param derivative Function pointer to the gradient operation.
     * @param A_ptr Pointer to the first input node.
     * @param B_ptr Pointer to the second input node.
     * @param tensor_args Arguments to construct the output tensor.
     */
    Node_dot(INode *A_ptr, INode *B_ptr);

    /**
     * @brief Evaluates the binary operation by applying forward_op, if not
     * already evaluated.
     */
    void eval() override;

    /**
     * @brief Propagates gradients back through the binary operation by applying
     * backward_op.
     */
    void getGrad() override;
};

} // namespace kaad
