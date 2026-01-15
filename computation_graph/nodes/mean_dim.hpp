#pragma once

#include "../../tensorfuncs/adjoint_ops.hpp" // for tensorfuncs::adjoint
#include "../../tensorfuncs/primal_ops.hpp"  // for tensorfuncs::primal
#include "inode.hpp"                         // for INode

namespace kaad {

/**
 * @brief A mean_dim operation node in a computation graph.
 *
 * Applies a mean_dim operation during forward evaluation and its corresponding
 * gradient function during backpropagation.
 *
 * @tparam T The scalar type.
 */
template <typename T> struct Node_mean_dim : INode<T> {
    tensorfuncs::primal::unary::mean_dim_fn<T> val_func =
        tensorfuncs::primal::unary::mean_dim;
    tensorfuncs::adjoint::unary::mean_dim_fn<T> grad_func =
        tensorfuncs::adjoint::unary::mean_dim;

    int *strideA = nullptr; ///< stride Array for A.
    int *strideC = nullptr; ///< stride Array for C.
    size_t *A_offset =
        nullptr; ///< Total number of elements and per-dim offsets of A.
    const T *C_end =
        nullptr; ///< Pointer to the end of the C buffer (used for iteration).
    const T *dA_end =
        nullptr;  ///< Pointer to the end of the dA buffer (used for iteration).
    size_t D = 0; ///< Number of the dimensions of the A tensor.
    T divisor = 0; ///< Divisor to compute the mean of the A tensor (length of A
                   ///< buffer).

    /**
     * @brief Constructs a mean_dim node with the given operation and gradient.
     *
     * @param A_ptr    Pointer to the input node.
     * @param args       Arguments to construct the output tensor.
     */
    template <typename... Args>
    Node_mean_dim(INode<T> *A_ptr, Args &&...args) : INode<T>(A_ptr, args...) {}

    /**
     * @brief Destructor for Node_mean_dim.
     *
     * Frees dynamically allocated memory for stride and offset arrays.
     */
    ~Node_mean_dim() {
        delete[] strideA;
        delete[] strideC;
        delete[] A_offset;
    }

    /**
     * @brief Evaluates the mean_dim operation if not already evaluated.
     *
     * Calls eval on the input node and applies `val_func` to compute this
     * node's value.
     */
    inline void eval() override {
        if (!this->evaluated) {
            this->A->eval();

            val_func(this->A->value.data(), this->value.elements_.data(),
                     strideA, strideC, A_offset, D, divisor, C_end);
            this->evaluated = true;
        }
    }

    /**
     * @brief Propagates gradients back through the mean_dim operation.
     *
     * Applies `grad_func` to compute input gradients and recursively calls/
     * `getGrad` on the input node if it has further dependencies.
     */
    inline void getGrad() override {
        grad_func(this->A->value.data(), this->A->gradient.elements_.data(),
                  this->value.data(), this->gradient.data(), strideA, strideC,
                  A_offset, D, divisor, dA_end);

        if (this->A->hasInputs) {
            this->A->getGrad();
        }
    }
};

} // namespace kaad
