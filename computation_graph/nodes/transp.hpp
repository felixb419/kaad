#pragma once

#include "../../tensorfuncs/adjoint_ops.hpp" // for tensorfuncs::adjoint
#include "../../tensorfuncs/kernels.hpp"     // for NoOp
#include "../../tensorfuncs/primal_ops.hpp"  // for tensorfuncs::primal
#include "inode.hpp"                         // for INode

namespace kaad {

/**
 * @brief A transpose operation node in a computation graph.
 * @see tensorfuncs::primal::unary::noop
 * @see tensorfuncs::adjoint::unary::pointwise
 */
class Node_transp : public INode {
  public:
    const char *node_type() const noexcept override;

    tensorfuncs::primal::unary::pointwise_fn<Scalar, Kernels::NoOp<Scalar>>
        forward_op = tensorfuncs::primal::unary::pointwise<
            Scalar,
            Kernels::NoOp<Scalar>>; ///< Function pointer to the value
                                    ///< operation.

    tensorfuncs::adjoint::unary::pointwise_fn<Scalar, Kernels::NoOp<Scalar>>
        backward_op = tensorfuncs::adjoint::unary::pointwise<
            Scalar,
            Kernels::NoOp<Scalar>>; ///< Function pointer to the gradient
                                    ///< operation.

    const Scalar *A_end = nullptr; ///< Pointer to the end of the A buffer.
    const Scalar *C_end = nullptr; ///< Pointer to the end of the C buffer.

    /**
     * @brief Constructs a transpose node with the given operation and gradient.
     *
     * @param A_ptr Pointer to the input node.
     * @param tensor_args Arguments to construct the output tensor.
     */
    template <typename... TensorArgs>
    Node_transp(INode *A_ptr, TensorArgs &&...tensor_args)
        : INode(A_ptr, tensor_args...) {
        INode *base_ptr = static_cast<INode *>(this);
        this->A_end = base_ptr->A->value.data() + base_ptr->A->value.size();
        this->C_end = base_ptr->value.data() + base_ptr->value.size();
    }

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
