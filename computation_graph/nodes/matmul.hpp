#pragma once

#include "../../tensorfuncs/adjoint_ops.hpp" // for tensorfuncs::adjoint
#include "../../tensorfuncs/primal_ops.hpp"  // for tensorfuncs::primal
#include "inode.hpp"                         // for INode

namespace kaad {

/**
 * @brief A matmul node in a computation graph.
 *
 * Applies a matmul operation to two tensors with compatible shapes during
 * forward evaluation and its corresponding gradient function during
 * backpropagation.
 *
 * @tparam T The scalar type.
 */
template <typename T> struct Node_matmul : INode<T> {
    INode<T> *B = nullptr; ///< Pointer to the second input Node.

    tensorfuncs::primal::binary::matmul_fn<T> val_func =
        tensorfuncs::primal::binary::matmul; ///< Function pointer to the matmul
                                             ///< operation.
    tensorfuncs::adjoint::binary::matmul_fn<T> grad_func =
        tensorfuncs::adjoint::binary::matmul; ///< Function pointer to the
                                              ///< matmul gradient.

    /**
     * @brief Stride arrays for tensors A, B, and C for all computation stages.
     *
     * Each stride array is a flattened array of size 6, containing strides for
     * all 3 passes (forward and both backward gradients).
     *
     * Index layout for each stride array:
     * - [0..1] Forward pass (C = A * B)
     * - [2..3] Gradient w.r.t. A (dA = dC * Bᵗ)
     * - [4..5] Gradient w.r.t. B (dB = Aᵗ * dC)
     *
     * Each pair of entries represents strides for rows and columns, or outer
     * and inner dimensions, depending on layout.
     */
    int a_dim[3]; ///< Number of rows of tensor A for each computation stage.
    int b_dim[3]; ///< Number of columns of tensor B for each computation stage.
    int k[3];     ///< Shared inner dimension for each computation stage.
    int strideA[6]; ///< Flattened stride pairs for tensor A (2 per stage × 3
                    ///< stages).
    int strideB[6]; ///< Flattened stride pairs for tensor B (2 per stage × 3
                    ///< stages).
    int strideC[6]; ///< Flattened stride pairs for tensor C (2 per stage × 3
                    ///< stages).

    /**
     * @brief Constructs a matmul node.
     *
     * @param A_ptr Pointer to the first input node.
     * @param B_ptr Pointer to the second input node.
     * @param args Arguments to construct the output tensor.
     */
    template <typename... Args>
    Node_matmul(INode<T> *A_ptr, INode<T> *B_ptr, Args &&...args)
        : B(B_ptr), INode<T>(A_ptr, args...) {}

    /**
     * @brief Evaluates the matmul operation if not already evaluated.
     *
     * Calls eval on the input nodes and applies `val_func` to compute this
     * node's value.
     */
    inline void eval() override {
        if (!this->evaluated) {
            this->A->eval();
            this->B->eval();

            val_func(this->A->value.data(), this->B->value.data(),
                     this->value.data(), a_dim[0], b_dim[0], k[0], strideA,
                     strideB, strideC);
            this->evaluated = true;
        }
    }

    /**
     * @brief Propagates gradients back through the matmul operation.
     *
     * Applies `grad_func` to compute input gradients and recursively calls
     * `getGrad` on the input nodes if they have further dependencies.
     */
    inline void getGrad() override {
        grad_func(this->A->value.data(), this->A->gradient.data(),
                  this->B->value.data(), this->B->gradient.data(),
                  this->value.data(), this->gradient.data(), a_dim + 1,
                  b_dim + 1, k + 1, strideA + 2, strideB + 2, strideC + 2);

        if (this->A->hasInputs) {
            this->A->getGrad();
        }
        if (this->B->hasInputs) {
            this->B->getGrad();
        }
    }
};

} // namespace kaad
