#pragma once

#include "../../tensorfuncs/adjoint_ops.hpp" // for tensorfuncs::adjoint
#include "../../tensorfuncs/primal_ops.hpp"  // for tensorfuncs::primal
#include "inode.hpp"                         // for INode

namespace kaad {

/**
 * @brief A slice operation node in a computation graph.
 *
 * Applies a slice operation during forward evaluation and its corresponding
 * gradient function during backpropagation.
 *
 * @tparam T The scalar type.
 */
template <typename T> struct Node_slice : INode<T> {
    tensorfuncs::primal::unary::slice_fn<T> val_func =
        tensorfuncs::primal::unary::slice<T>;
    tensorfuncs::adjoint::unary::slice_fn<T> grad_func =
        tensorfuncs::adjoint::unary::slice<T>;

    int *strideA = nullptr;           ///< Stride array for tensor A.
    int *strideB = nullptr;           ///< Stride array for tensor B.
    int *strideC = nullptr;           ///< Stride array for tensor C.
    size_t *start_offset_a = nullptr; ///< Offset for the start of A.
    size_t *C_offset = nullptr; ///< Per-dim offset to the end of C buffer.
    size_t D = 0;               ///< Number of the dimensions of the C tensor.

    /**
     * @brief Constructs a slice node.
     *
     * @param A_ptr    Pointer to the input node.
     * @param args       Arguments to construct the output tensor.
     */
    template <typename... Args>
    Node_slice(INode<T> *A_ptr, Args &&...args) : INode<T>(A_ptr, args...) {}

    /**
     * @brief Destructor for Node_slice.
     *
     * Frees dynamically allocated memory for stride and offset arrays.
     */
    ~Node_slice() {
        delete[] strideA;
        delete[] strideB;
        delete[] strideC;
        delete[] C_offset;
    }

    /**
     * @brief Evaluates the slice operation if not already evaluated.
     *
     * Calls eval on the input node and applies `val_func` to compute this
     * node's value.
     */
    inline void eval() override {
        if (!this->evaluated) {
            this->A->eval();

            val_func(this->A->value.data(), this->value.elements.data(),
                     strideA, strideC, start_offset_a, C_offset, D);
            this->evaluated = true;
        }
    }

    /**
     * @brief Propagates gradients back through the slice operation.
     *
     * Applies `grad_func` to compute input gradients and recursively calls/
     * `getGrad` on the input node if it has further dependencies.
     */
    inline void getGrad() override {
        grad_func(this->A->gradient.elements.data(), this->gradient.data(),
                  strideA, strideC, start_offset_a, C_offset, D);

        if (this->A->hasInputs) {
            this->A->getGrad();
        }
    }
};

} // namespace kaad
