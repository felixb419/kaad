#pragma once

#include "../../tensorfuncs/adjoint_ops.hpp" // for tensorfuncs::adjoint
#include "../../tensorfuncs/primal_ops.hpp"  // for tensorfuncs::primal
#include "inode.hpp"                         // for INode

namespace kaad {

/**
 * @brief A mean operation node in a computation graph.
 * @see tensorfuncs::primal::unary::mean
 * @see tensorfuncs::adjoint::unary::mean
 */
class Node_mean : public INode {
  public:
    const char *node_type() const noexcept override;

    tensorfuncs::primal::unary::mean_fn<Scalar> forward_op =
        tensorfuncs::primal::unary::mean;
    tensorfuncs::adjoint::unary::mean_fn<Scalar> backward_op =
        tensorfuncs::adjoint::unary::mean;

    const Scalar *A_end =
        nullptr; ///< Pointer to the end of the A buffer (used for iteration).
    const Scalar *dA_end =
        nullptr; ///< Pointer to the end of the dA buffer (used for iteration).
    Scalar divisor = 0; ///< Divisor to compute the mean of the A tensor (length
                        ///< of A buffer).

    /**
     * @brief Constructs a mean node.
     * @param A_ptr Pointer to the input node.
     * @param tensor_args Arguments to construct the output tensor.
     */
    template <typename... TensorArgs>
    Node_mean(INode *A_ptr, TensorArgs &&...tensor_args)
        : INode(A_ptr, tensor_args...) {
        this->A_end = A_ptr->value.data() + A_ptr->value.size();
        this->dA_end = A_ptr->gradient.data() + A_ptr->gradient.size();
        this->divisor = A_ptr->value.size();
    }

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
