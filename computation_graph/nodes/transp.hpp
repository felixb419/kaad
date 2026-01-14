#pragma once

#include "../../tensorfuncs/adjoint_ops.hpp" // for tensorfuncs::adjoint
#include "../../tensorfuncs/kernels.hpp"     // for Null::Op
#include "../../tensorfuncs/primal_ops.hpp"  // for tensorfuncs::primal
#include "inode.hpp"                         // for INode

namespace kaad {

/**
 * @brief A transpose operation node in a computation graph.
 *
 * Applies a transpose operation during forward evaluation and its corresponding
 * gradient function during backpropagation.
 *
 * @tparam T The scalar type.
 */
template <typename T> struct Node_transp : INode<T> {
    using Op = typename Kernels::Null::Op;
    Op op;
    tensorfuncs::primal::unary::pointwise_fn<T, Op> val_func =
        tensorfuncs::primal::unary::noop<T>; ///< Function pointer to the
                                             ///< value operation.

    using Grad = typename Kernels::Sum<T>::Grad;
    Grad grad;
    tensorfuncs::adjoint::unary::pointwise_fn<T, Grad> grad_func =
        tensorfuncs::adjoint::unary::pointwise<T, Grad>; ///< Function pointer
                                                         ///< to the gradient
                                                         ///< operation.

    const T *A_end = nullptr; ///< Pointer to the end of the A buffer.
    const T *C_end = nullptr; ///< Pointer to the end of the C buffer.

    /**
     * @brief Constructs a transpose node with the given operation and gradient.
     *
     * @param A_ptr    Pointer to the input node.
     * @param args       Arguments to construct the output tensor.
     */
    template <typename... Args>
    Node_transp(INode<T> *A_ptr, Args &&...args) : INode<T>(A_ptr, args...) {}

    /**
     * @brief Evaluates the transpose operation if not already evaluated.
     *
     * Calls eval on the input node and applies `val_func` to compute this
     * node's value.
     */
    inline void eval() override {
        if (!this->evaluated) {
            this->A->eval();

            val_func(this->A->value.data(), this->value.elements.data(), A_end,
                     op);
            this->evaluated = true;
        }
    }

    /**
     * @brief Propagates gradients back through the transpose operation.
     *
     * Applies `grad_func` to compute input gradients and recursively calls/
     * `getGrad` on the input node if it has further dependencies.
     */
    inline void getGrad() override {
        grad_func(this->A->value.data(), this->A->gradient.elements.data(),
                  this->value.data(), this->gradient.data(), C_end, grad);

        if (this->A->hasInputs) {
            this->A->getGrad();
        }
    }
};

} // namespace kaad
