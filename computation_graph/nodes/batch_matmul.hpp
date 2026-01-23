#pragma once

#include "../../tensorfuncs/adjoint_ops.hpp" // for tensorfuncs::adjoint
#include "../../tensorfuncs/primal_ops.hpp"  // for tensorfuncs::primal
#include "../strides.hpp"                    // for Strides::batch_matmul
#include "../dispatchers.hpp" // for get_batch_matmul, get_batch_matmul_grad
#include "inode.hpp"       // for INode

namespace kaad {

/**
 * @brief A batch_matmul operation node in a computation graph.
 * @see tensorfuncs::primal::binary::batch_matmul
 * @see tensorfuncs::adjoint::binary::batch_matmul
 * @tparam T The scalar type.
 * @tparam Kernel A kernel struct providing `Op` and `Grad` types for the
 * operation.
 */
template <typename T> class Node_batch_matmul : public INode<T> {
  public:
    INode<T> *B = nullptr; ///< Pointer to the second input Node.

    tensorfuncs::primal::binary::batch_matmul_fn<T> forward_op =
        tensorfuncs::primal::binary::batch_matmul; ///< Function pointer to the
                                                   ///< batch_matmul operation.
    tensorfuncs::adjoint::binary::batch_matmul_fn<T> backward_op =
        tensorfuncs::adjoint::binary::batch_matmul; ///< Function pointer to the
                                                    ///< batch_matmul gradient.

    /**
     * @brief Stride arrays for A, B, and C for each stage of computation.
     * Index convention:
     * - [0] Forward pass (C = A * B)
     * - [1] Gradient w.r.t. A (dA = dC * Bᵗ)
     * - [2] Gradient w.r.t. B (dB = Aᵗ * dC)
     */
    int *(strideA[3]);  ///< Stride array for tensor A.
    int *(strideB[3]);  ///< Stride array for tensor B.
    int *(strideC[3]);  ///< Stride array for tensor C.
    int *(c_shape[3]);  ///< shape of C (used for iteration).
    int A_colStride[3]; ///< Gap between columns of the A matrix.
    int B_rowStride[3]; ///< Gap between rows of the B matrix.
    int shared_dim[3];  ///< Shared inner dimension of the tensors.
    size_t C_nDims = 0; ///< Number of the dimensions of the value tensor.

    /**
     * @brief Constructs a batch_matmul node.
     * @param A_ptr Pointer to the first input node.
     * @param B_ptr Pointer to the second input node.
     * @param tensor_args Arguments to construct the output tensor.
     */
    template <typename... TensorArgs>
    Node_batch_matmul(INode<T> *A_ptr, INode<T> *B_ptr,
                      TensorArgs &&...tensor_args)
        : B(B_ptr), INode<T>(A_ptr, tensor_args...) {
        Strides::batch_matmul<T>(*this);

        if (C_nDims <= Dispatchers::MAX_NDIMS) {
            forward_op = Dispatchers::get_batch_matmul<T>()[C_nDims];
            backward_op = Dispatchers::get_batch_matmul_grad<T>()[C_nDims];
        }
    }

    /**
     * @brief Destructor for Node_batch_matmul.
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
     * @brief Evaluates the batch_matmul operation by calling forward_op, if not
     * already evaluated.
     */
    inline void eval() override {
        if (!this->evaluated) {
            this->A->eval();
            this->B->eval();

            forward_op(this->A->value.data(), this->B->value.data(),
                       this->value.elements_.data(), strideA[0], strideB[0],
                       strideC[0], c_shape[0], A_colStride[0], B_rowStride[0],
                       shared_dim[0], C_nDims);
            this->evaluated = true;
        }
    }

    /**
     * @brief Propagates gradients back through batch batch_matmul operation by
     * calling backward_func.
     */
    inline void getGrad() override {
        backward_op(this->A->value.data(), this->A->gradient.elements_.data(),
                    this->B->value.data(), this->B->gradient.elements_.data(),
                    this->value.data(), this->gradient.data(), strideA + 1,
                    strideB + 1, strideC + 1, c_shape + 1, A_colStride + 1,
                    B_rowStride + 1, shared_dim + 1, C_nDims);

        if (this->A->hasInputs) {
            this->A->getGrad();
        }
        if (this->B->hasInputs) {
            this->B->getGrad();
        }
    }
};

} // namespace kaad
