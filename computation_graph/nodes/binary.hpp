#pragma once

#include "../../tensorfuncs/adjoint_ops.hpp" // for tensorfuncs::adjoint
#include "../../tensorfuncs/primal_ops.hpp"  // for tensorfuncs::primal
#include "inode.hpp"                         // for INode

namespace kaad {

/**
 * @brief A binary operation node in a computation graph.
 * @see tensorfuncs::primal::binary::pointwise
 * @see tensorfuncs::adjoint::binary::pointwise
 * @tparam T The scalar type.
 * @tparam Kernel A kernel struct providing `Op` and `Grad` types for the
 * operation.
 */
template <typename T, class Kernel> struct Node_binary : INode<T> {
    INode<T> *B = nullptr; ///< Pointer to the second input Node.

    using Op = class Kernel::Op; ///< Type alias for the operation kernel.
    Op op;

    tensorfuncs::primal::binary::pointwise_fn<T, Op> forward_op =
        tensorfuncs::primal::binary::pointwise; ///< Function pointer to the
                                                ///< value operation.

    using Grad = class Kernel::Grad; ///< Type alias for the gradient Kernel.
    Grad grad;

    tensorfuncs::adjoint::binary::pointwise_fn<T, Grad> backward_op =
        tensorfuncs::primal::binary::pointwise; ///< Function pointer to the
                                                ///< gradient operation.

    const T *C_end = nullptr; ///< Pointer to the end of the value buffer (used
                              ///< for iteration).

    /**
     * @brief Constructs a binary operation node with the given operation and
     * gradient.
     * @param operation Function pointer to the value operation.
     * @param derivative Function pointer to the gradient operation.
     * @param A_ptr Pointer to the first input node.
     * @param B_ptr Pointer to the second input node.
     * @param tensor_args Arguments to construct the output tensor.
     */
    template <typename... TensorArgs>
    Node_binary(tensorfuncs::primal::binary::pointwise_fn<T, Op> operation,
                tensorfuncs::adjoint::binary::pointwise_fn<T, Grad> derivative,
                INode<T> *A_ptr, INode<T> *B_ptr, TensorArgs &&...tensor_args)
        : B(B_ptr), forward_op(operation), backward_op(derivative),
          INode<T>(A_ptr, tensor_args...) {}

    /**
     * @brief Evaluates the binary operation by applying forward_op, if not
     * already evaluated.
     */
    inline void eval() override {
        if (!this->evaluated) {
            this->A->eval();
            this->B->eval();

            forward_op(this->A->value.data(), this->B->value.data(),
                       this->value.elements_.data(), C_end, op);
            this->evaluated = true;
        }
    }

    /**
     * @brief Propagates gradients back through the binary operation by applying
     * backward_op.
     */
    inline void getGrad() override {
        backward_op(this->A->value.data(), this->A->gradient.elements_.data(),
                    this->B->value.data(), this->B->gradient.elements_.data(),
                    this->value.data(), this->gradient.data(), C_end, grad);

        if (this->A->hasInputs) {
            this->A->getGrad();
        }
        if (this->B->hasInputs) {
            this->B->getGrad();
        }
    }
};

} // namespace kaad
