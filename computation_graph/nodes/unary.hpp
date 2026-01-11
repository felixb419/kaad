#pragma once

#include "../../tensorfuncs/adjoint_ops.hpp" // for tensorfuncs::adjoint
#include "../../tensorfuncs/primal_ops.hpp"  // for tensorfuncs::primal
#include "inode.hpp"                         // for INode

namespace kaad {

/**
 * @brief A unary operation node in a computation graph.
 *
 * Applies a unary operation during forward evaluation and its corresponding
 * gradient function during backpropagation.
 *
 * @tparam T The scalar type.
 * @tparam Kernel A kernel struct providing `Op` and `Grad` types for the
 * operation.
 */
template <typename T, class Kernel> struct Node_unary : INode<T> {
    using Op = class Kernel::Op; ///< Type alias for the operation kernel.
    Op op;

    tensorfuncs::primal::unary::pointwise_fn<T, Op> val_func =
        nullptr; ///< Function pointer to the value operation.

    using Grad = class Kernel::Grad; ///< Type alias for the gradient kernel.
    Grad grad;

    tensorfuncs::adjoint::unary::pointwise_fn<T, Grad> grad_func =
        nullptr; ///< Function pointer to the gradient operation.

    T *end = nullptr; ///< Pointer to the end of the value buffer (used for
                      ///< iteration)

    /**
     * @brief Constructs a unary node with the given operation and gradient.
     *
     * @param operation  Function pointer to the value operation.
     * @param derivative Function pointer to the gradient operation.
     * @param A_ptr    Pointer to the input node.
     * @param args       Arguments to construct the output tensor.
     */
    template <typename... Args>
    Node_unary(tensorfuncs::primal::unary::pointwise_fn<T, Op> operation,
               tensorfuncs::adjoint::unary::pointwise_fn<T, Grad> derivative,
               INode<T> *A_ptr, Args &&...args)
        : val_func(operation), grad_func(derivative), INode<T>(A_ptr, args...) {
    }

    /**
     * @brief Evaluates the unary operation if not already evaluated.
     *
     * Calls eval on the input node and applies `val_func` to compute this
     * node's value.
     */
    inline void eval() override {
        if (!this->evaluated) {
            this->A->eval();

            val_func(this->A->value.data(), this->value.data(), end, op);
            this->evaluated = true;
        }
    }

    /**
     * @brief Propagates gradients back through the unary operation.
     *
     * Applies `grad_func` to compute input gradients and recursively calls/
     * `getGrad` on the input node if it has further dependencies.
     */
    inline void getGrad() override {
        grad_func(this->A->value.data(), this->A->gradient.data(),
                  this->value.data(), this->gradient.data(), end, grad);

        if (this->A->hasInputs) {
            this->A->getGrad();
        }
    }
};

} // namespace kaad
