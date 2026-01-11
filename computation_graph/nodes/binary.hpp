#pragma once

#include "../../tensorfuncs/adjoint_ops.hpp" // for tensorfuncs::adjoint
#include "../../tensorfuncs/primal_ops.hpp"  // for tensorfuncs::primal
#include "inode.hpp"                         // for INode

namespace kaad {

/**
 * @brief A binary operation node in a computation graph.
 *
 * Applies a binary operation to two tensors with matching shapes during forward
 * evaluation and its corresponding gradient function during backpropagation.
 *
 * @tparam T The scalar type.
 * @tparam Kernel A kernel struct providing `Op` and `Grad` types for the
 * operation.
 */
template <typename T, class Kernel> struct Node_binary : INode<T> {
    INode<T> *B = nullptr; ///< Pointer to the second input Node.

    using Op = class Kernel::Op; ///< Type alias for the operation kernel.
    Op op;

    tensorfuncs::primal::binary::pointwise_fn<T, Op> val_func =
        nullptr; ///< Function pointer to the value operation.

    using Grad = class Kernel::Grad; ///< Type alias for the gradient Kernel.
    Grad grad;

    tensorfuncs::adjoint::binary::pointwise_fn<T, Grad> grad_func =
        nullptr; ///< Function pointer to the gradient operation.

    T *end = nullptr; ///< Pointer to the end of the value buffer (used for
                      ///< iteration).

    /**
     * @brief Constructs a binary operation node with the given operation and
     * gradient.
     *
     * @param operation Function pointer to the value operation.
     * @param derivative Function pointer to the gradient operation.
     * @param A_ptr Pointer to the first input node.
     * @param B_ptr Pointer to the second input node.
     * @param args Arguments to construct the output tensor.
     */
    template <typename... Args>
    Node_binary(tensorfuncs::primal::binary::pointwise_fn<T, Op> operation,
                tensorfuncs::adjoint::binary::pointwise_fn<T, Grad> derivative,
                INode<T> *A_ptr, INode<T> *B_ptr, Args &&...args)
        : B(B_ptr), val_func(operation), grad_func(derivative),
          INode<T>(A_ptr, args...) {}

    /**
     * @brief Evaluates the binary operation if not already evaluated.
     *
     * Calls eval on the input nodes and applies `val_func` to compute this
     * node's value.
     */
    inline void eval() override {
        if (!this->evaluated) {
            this->A->eval();
            this->B->eval();

            val_func(this->A->value.data(), B->value.data(), this->value.data(),
                     end, op);
            this->evaluated = true;
        }
    }

    /**
     * @brief Propagates gradients back through the binary operation.
     *
     * Applies `grad_func` to compute input gradients and recursively calls
     * `getGrad` on the input nodes if they have further dependencies.
     */
    inline void getGrad() override {
        grad_func(this->A->value.data(), this->A->gradient.data(),
                  B->value.data(), B->gradient.data(), this->value.data(),
                  this->gradient.data(), end, grad);

        if (this->A->hasInputs) {
            this->A->getGrad();
        }
        if (this->B->hasInputs) {
            this->B->getGrad();
        }
    }
};

} // namespace kaad
