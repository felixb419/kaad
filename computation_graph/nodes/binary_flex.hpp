#pragma once

#include "../../tensorfuncs/adjoint_ops.hpp" // for tensorfuncs::adjoint
#include "../../tensorfuncs/primal_ops.hpp"  // for tensorfuncs::primal
#include "inode.hpp"                         // for INode

namespace kaad {

/**
 * @brief A binary_flex operation node in a computation graph.
 *
 * Applies a binary_flex operation to two tensors with broadcastable shapes
 * during forward evaluation and its corresponding gradient function during
 * backpropagation.
 *
 * @tparam T The scalar type.
 * @tparam Kernel A kernel struct providing `Op` and `Grad` types for the
 * operation.
 */
template <typename T, class Kernel> struct Node_binary_flex : INode<T> {
    INode<T> *B = nullptr; ///< Pointer to the second input Node.

    using Op = class Kernel::Op; ///< Type alias for the operation kernel.
    Op op;

    tensorfuncs::primal::binary::flexible_fn<T, Op> val_func =
        tensorfuncs::primal::binary::flexible<T, Op>; ///< Function pointer to
                                                      ///< the value operation.

    using Grad = class Kernel::Grad; ///< Type alias for the gradient kernel.
    Grad grad;

    tensorfuncs::adjoint::binary::flexible_fn<T, Grad> grad_func =
        tensorfuncs::adjoint::binary::flexible<T, Grad>; ///< Function pointer
                                                         ///< to the gradient
                                                         ///< operation.

    int *strideA = nullptr;     ///< stride Array for A.
    int *strideB = nullptr;     ///< stride Array for B.
    int *strideC = nullptr;     ///< stride Array for C.
    size_t *C_offset = nullptr; ///< Per-dim offset to the end of C buffer.
    size_t D = 0;               ///< Number of the dimensions of the C tensor.

    /**
     * @brief Constructs a binary_flex operation node with binary_flex operation
     * and gradient.
     *
     * @param A_ptr Pointer to the first input node.
     * @param B_ptr Pointer to the second input node.
     * @param args Arguments to construct the output tensor.
     */
    template <typename... Args>
    Node_binary_flex(INode<T> *A_ptr, INode<T> *B_ptr, Args &&...args)
        : B(B_ptr), INode<T>(A_ptr, args...) {}

    /**
     * @brief Destructor for Node_binary_flex.
     *
     * Frees dynamically allocated memory for stride and offset arrays.
     */
    ~Node_binary_flex() {
        delete[] strideA;
        delete[] strideB;
        delete[] strideC;
        delete[] C_offset;
    }

    /**
     * @brief Evaluates the binary_flex operation if not already evaluated.
     *
     * Calls eval on the input nodes and applies `val_func` to compute this
     * node's value.
     */
    inline void eval() override {
        if (!this->evaluated) {
            this->A->eval();
            this->B->eval();

            val_func(this->A->value.data(), this->B->value.data(),
                     this->value.elements.data(), strideA, strideB, strideC,
                     C_offset, D, op);
            this->evaluated = true;
        }
    }

    /**
     * @brief Propagates gradients back through the binary_flex operation.
     *
     * Applies `grad_func` to compute input gradients and recursively calls
     * `getGrad` on the input nodes if they have further dependencies.
     */
    inline void getGrad() override {
        grad_func(this->A->value.data(), this->A->gradient.elements.data(),
                  this->B->value.data(), this->B->gradient.elements.data(),
                  this->value.data(), this->gradient.data(), strideA, strideB,
                  strideC, C_offset, D, grad);

        if (this->A->hasInputs) {
            this->A->getGrad();
        }
        if (this->B->hasInputs) {
            this->B->getGrad();
        }
    }
};

} // namespace kaad
