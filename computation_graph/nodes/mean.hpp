#pragma once

#include "../../tensorfuncs/adjoint_ops.hpp" // for tensorfuncs::adjoint
#include "../../tensorfuncs/primal_ops.hpp"  // for tensorfuncs::primal
#include "inode.hpp"                         // for INode

namespace kaad {

/**
 * @brief A mean operation node in a computation graph.
 *
 * Applies the mean operation during forward evaluation and computes its
 * corresponding gradient during backpropagation.
 *
 * @tparam T The scalar type.
 */
template <typename T> struct Node_mean : INode<T> {
    tensorfuncs::primal::unary::mean_fn<T> val_func =
        tensorfuncs::primal::unary::mean;
    tensorfuncs::adjoint::unary::mean_fn<T> grad_func =
        tensorfuncs::adjoint::unary::mean;

    const T *A_end =
        nullptr; ///< Pointer to the end of the A buffer (used for iteration).
    const T *dA_end =
        nullptr; ///< Pointer to the end of the dA buffer (used for iteration).
    T divisor = 0; ///< Divisor to compute the mean of the A tensor (length of A
                   ///< buffer).

    /**
     * @brief Constructs a mean node.
     *
     * @param A_ptr Pointer to the input node.
     * @param args Arguments to construct the output tensor.
     */
    template <typename... Args>
    Node_mean(INode<T> *A_ptr, Args &&...args) : INode<T>(A_ptr, args...) {}

    /**
     * @brief Evaluates the mean operation if not already evaluated.
     *
     * Calls eval on the input node and applies `val_func` to compute this
     * node's value.
     */
    inline void eval() override {
        if (!this->evaluated) {
            this->A->eval();

            val_func(this->A->value.data(), this->value.elements.data(), A_end,
                     divisor);
            this->evaluated = true;
        }
    }

    /**
     * @brief Propagates gradients back through the mean operation.
     *
     * Applies `grad_func` to compute input gradients and recursively calls/
     * `getGrad` on the input node if it has further dependencies.
     */
    inline void getGrad() override {
        grad_func(this->A->gradient.elements.data(), this->gradient.data(),
                  dA_end, divisor);

        if (this->A->hasInputs) {
            this->A->getGrad();
        }
    }
};

} // namespace kaad
