#pragma once

#include "../../tensorfuncs/adjoint_ops.hpp" // for tensorfuncs::adjoint
#include "../../tensorfuncs/primal_ops.hpp"  // for tensorfuncs::primal
#include "inode.hpp"                         // for INode

namespace kaad {

/**
 * @brief A sum_dim operation node in a computation graph.
 *
 * Applies a sum_dim operation during forward evaluation and its
 * corresponding gradient function during backpropagation.
 *
 * @tparam T The scalar type.
 */
template <typename T> struct Node_sum_dim : INode<T> {
    tensorfuncs::primal::unary::sum_dim_fn<T> val_func =
        tensorfuncs::primal::unary::sum_dim; ///< Function pointer to the
                                             ///< sum_dim operation.
    tensorfuncs::adjoint::unary::sum_dim_fn<T> grad_func =
        tensorfuncs::adjoint::unary::sum_dim; ///< Function pointer to the
                                              ///< sum_dim gradient.

    int *strideA = nullptr;     ///< stride Array for A.
    int *strideC = nullptr;     ///< stride Array for C.
    size_t *A_offset = nullptr; ///< Per-dim offset to the end of A buffer.
    size_t D = 0;               ///< Number of dimensions of A

    /**
     * @brief Constructs a sum_dim node.
     *
     * @param A_ptr    Pointer to the input node.
     * @param args       Arguments to construct the output tensor.
     */
    template <typename... Args>
    Node_sum_dim(INode<T> *A_ptr, Args &&...args) : INode<T>(A_ptr, args...) {}

    /**
     * @brief Destructor for Node_sum_dim.
     *
     * Frees dynamically allocated memory for stride and offset arrays.
     */
    ~Node_sum_dim() {
        delete[] strideA;
        delete[] strideC;
        delete[] A_offset;
    }

    /**
     * @brief Evaluates the sum_dim operation if not already evaluated.
     *
     * Calls eval on the input node and applies `val_func` to compute this
     * node's value.
     */
    inline void eval() override {
        if (!this->evaluated) {
            this->A->eval();

            val_func(this->A->value.data(), this->value.elements_.data(),
                     strideA, strideC, A_offset, D);
            this->evaluated = true;
        }
    }

    /**
     * @brief Propagates gradients back through the sum_dim operation.
     *
     * Applies `grad_func` to compute input gradients and recursively calls/
     * `getGrad` on the input node if it has further dependencies.
     */
    inline void getGrad() override {
        grad_func(this->A->gradient.elements_.data(), this->gradient.data(),
                  strideA, strideC, A_offset, D);

        if (this->A->hasInputs) {
            this->A->getGrad();
        }
    }
};

} // namespace kaad
