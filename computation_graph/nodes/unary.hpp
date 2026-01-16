#pragma once

#include "../../tensorfuncs/adjoint_ops.hpp" // for tensorfuncs::adjoint
#include "../../tensorfuncs/primal_ops.hpp"  // for tensorfuncs::primal
#include "inode.hpp"                         // for INode

namespace kaad {

/**
 * @brief A unary operation node in a computation graph.
 * @see tensorfuncs::primal::unary::pointwise
 * @see tensorfuncs::adjoint::unary::pointwise
 * @tparam T The scalar type.
 * @tparam Kernel A kernel struct providing `Op` and `Grad` types for the
 * operation.
 */
template <typename T, class Kernel> struct Node_unary : INode<T> {
    using Op = class Kernel::Op; ///< Type alias for the operation kernel.
    Op op;

    tensorfuncs::primal::unary::pointwise_fn<T, Op> forward_op =
        tensorfuncs::primal::unary::pointwise; ///< Function pointer to the
                                               ///< value operation.

    using Grad = class Kernel::Grad; ///< Type alias for the gradient kernel.
    Grad grad;

    tensorfuncs::adjoint::unary::pointwise_fn<T, Grad> backward_op =
        tensorfuncs::adjoint::unary::pointwise; ///< Function pointer to the
                                                ///< gradient operation.

    const T *C_end = nullptr; ///< Pointer to the end of the value buffer (used
                              ///< for iteration)

    /**
     * @brief Constructs a unary node with the given operation and gradient.
     * @param operation  Function pointer to the value operation.
     * @param derivative Function pointer to the gradient operation.
     * @param A_ptr    Pointer to the input node.
     * @param tensor_args       Arguments to construct the output tensor.
     */
    template <typename... TensorArgs>
    Node_unary(tensorfuncs::primal::unary::pointwise_fn<T, Op> operation,
               tensorfuncs::adjoint::unary::pointwise_fn<T, Grad> derivative,
               INode<T> *A_ptr, TensorArgs &&...tensor_args)
        : forward_op(operation), backward_op(derivative),
          INode<T>(A_ptr, tensor_args...) {}

    /**
     * @brief Evaluates the unary operation by applying forward_op, if not
     * already evaluated.
     */
    inline void eval() override {
        if (!this->evaluated) {
            this->A->eval();

            forward_op(this->A->value.data(), this->value.elements_.data(),
                       C_end, op);
            this->evaluated = true;
        }
    }

    /**
     * @brief Propagates gradients back through the unary operation, by applying
     * backward_op.
     */
    inline void getGrad() override {
        backward_op(this->A->value.data(), this->A->gradient.elements_.data(),
                    this->value.data(), this->gradient.data(), C_end, grad);

        if (this->A->hasInputs) {
            this->A->getGrad();
        }
    }
};

} // namespace kaad
