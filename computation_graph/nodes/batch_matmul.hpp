#pragma once

#include "../../tensorfuncs/adjoint_ops.hpp" // for tensorfuncs::adjoint
#include "../../tensorfuncs/primal_ops.hpp"  // for tensorfuncs::primal
#include "inode.hpp"                         // for INode

namespace kaad {

/**
 * @brief A batch_matmul operation node in a computation graph.
 *
 * Applies a batch_matmul operation to two tensors with matching shapes during
 * forward evaluation and its corresponding gradient function during
 * backpropagation.
 *
 * @tparam T The scalar type.
 * @tparam Kernel A kernel struct providing `Op` and `Grad` types for the
 * operation.
 */
template <typename T> struct Node_batch_matmul : INode<T> {
    INode<T> *B = nullptr; ///< Pointer to the second input Node.

    tensorfuncs::primal::binary::batch_matmul_fn<T> val_func =
        tensorfuncs::primal::binary::batch_matmul; ///< Function pointer to the
                                                   ///< batch_matmul operation.
    tensorfuncs::adjoint::binary::batch_matmul_fn<T> grad_func =
        tensorfuncs::adjoint::binary::batch_matmul; ///< Function pointer to the
                                                    ///< batch_matmul gradient.

    /**
     * @brief Stride arrays for A, B, and C for each stage of computation.
     *
     * Index convention:
     * - [0] Forward pass (C = A * B)
     * - [1] Gradient w.r.t. A (dA = dC * Bᵗ)
     * - [2] Gradient w.r.t. B (dB = Aᵗ * dC)
     */
    int *strideA[3]; ///< Stride array for tensor A.
    int *strideB[3]; ///< Stride array for tensor B.
    int *strideC[3]; ///< Stride array for tensor C.
    int *c_shape[3]; ///< shape of C (used for iteration).
    int A_offset[3]; ///< Gap between columns of the A matrix.
    int B_offset[3]; ///< Gap between rows of the B matrix.
    int k[3];        ///< shared inner dimension of the tensors.
    size_t D = 0;    ///< Number of the dimensions of the C tensor.

    /**
     * @brief Constructs a batch_matmul node.
     *
     * @param A_ptr Pointer to the first input node.
     * @param B_ptr Pointer to the second input node.
     * @param args Arguments to construct the output tensor.
     */
    template <typename... Args>
    Node_batch_matmul(INode<T> *A_ptr, INode<T> *B_ptr, Args &&...args)
        : B(B_ptr), INode<T>(A_ptr, args...) {}

    /**
     * @brief Destructor for Node_batch_matmul.
     *
     * Frees dynamically allocated memory for stride and shape arrays.
     */
    ~Node_batch_matmul() {
        for (int i = 0; i < 3; i++) {
            delete[] strideA[i];
            delete[] strideB[i];
            delete[] strideC[i];
            delete[] c_shape[i];
        }
    }

    /**
     * @brief Evaluates the batch_matmul operation if not already
     * evaluated.
     *
     * Calls eval on the input nodes and applies `val_func` to compute this
     * node's value.
     */
    inline void eval() override {
        if (!this->evaluated) {
            this->A->eval();
            this->B->eval();

            val_func(this->A->value.data(), this->B->value.data(),
                     this->value.elements_.data(), strideA[0], strideB[0],
                     strideC[0], c_shape[0], A_offset[0], B_offset[0], k[0], D);
            this->evaluated = true;
        }
    }

    /**
     * @brief Propagates gradients back through batch batch_matmul operation.
     *
     * Applies `grad_func` to compute input gradients and recursively calls
     * `getGrad` on the input nodes if they have further dependencies.
     */
    inline void getGrad() override {
        grad_func(this->A->value.data(), this->A->gradient.elements_.data(),
                  this->B->value.data(), this->B->gradient.elements_.data(),
                  this->value.data(), this->gradient.data(), strideA + 1,
                  strideB + 1, strideC + 1, c_shape + 1, A_offset + 1,
                  B_offset + 1, k + 1, D);

        if (this->A->hasInputs) {
            this->A->getGrad();
        }
        if (this->B->hasInputs) {
            this->B->getGrad();
        }
    }
};

} // namespace kaad
