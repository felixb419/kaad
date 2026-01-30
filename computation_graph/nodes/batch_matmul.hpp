#pragma once

#include "../../scalar.hpp"                  // for Scalar
#include "../../tensorfuncs/adjoint_ops.hpp" // for tensorfuncs::adjoint
#include "../../tensorfuncs/primal_ops.hpp"  // for tensorfuncs::primal
#include "../common.hpp"                     // for combine_matrix
#include "../dispatchers.hpp" // for get_batch_matmul, get_batch_matmul_grad
#include "inode.hpp"          // for INode

namespace kaad {

namespace detail {

/**
 * @brief Initializes stride and shape metadata for batched matrix
 * multiplication.
 * @param A        View of input tensor A.
 * @param B        View of input tensor B.
 * @param C        View of output tensor C.
 * @param strideA  Output stride array for A.
 * @param strideB  Output stride array for B.
 * @param strideC  Output stride array for C.
 * @param c_shape  Output shape array for C.
 * @param a_off    Output offset for A in kernel loops.
 * @param b_off    Output offset for B in kernel loops.
 * @param k        Output inner dimension (A.cols == B.rows).
 * @param D        Output number of dimensions in broadcasted result.
 */
void batch_matmul_metadata_impl(Tensor_view &A, Tensor_view &B, Tensor_view &C,
                                int *&strideA, int *&strideB, int *&strideC,
                                int *&c_shape, int &a_off, int &b_off, int &k,
                                size_t &D) {
    a_off = A.stride[A.nDims - 1];
    b_off = B.stride[B.nDims - 2];
    k = A.shape[A.nDims - 1];

    D = std::max(A.nDims, B.nDims);
    c_shape = new int[D];

    detail::combine_matrix(A.shape, A.nDims, B.shape, B.nDims, c_shape, D);

    strideA = new int[D];
    strideB = new int[D];
    strideC = new int[D];

    int idx, idxA, idxB, idxC;
    for (int i = 1; i <= D; i++) {
        idx = D - i;
        idxA = A.nDims - i;
        strideA[idx] = idxA >= 0 ? A.stride[idxA] : 0;
        idxB = B.nDims - i;
        strideB[idx] = idxB >= 0 ? B.stride[idxB] : 0;
        idxC = C.nDims - i;
        strideC[idx] = idxC >= 0 ? C.stride[idxC] : 0;
    }

    strideA[D - 1] = 0;
    strideB[D - 2] = 0;
}

} // namespace detail

/**
 * @brief A batch_matmul operation node in a computation graph.
 * @see tensorfuncs::primal::binary::batch_matmul
 * @see tensorfuncs::adjoint::binary::batch_matmul
 * @tparam Kernel A kernel struct providing `Op` and `Grad` types for the
 * operation.
 */
class Node_batch_matmul : public INode {
  public:
    const char *node_type() const noexcept override {
        return "Node_batch_matmul";
    }

    INode *B = nullptr; ///< Pointer to the second input Node.

    tensorfuncs::primal::binary::batch_matmul_fn<Scalar> forward_op =
        tensorfuncs::primal::binary::batch_matmul; ///< Function pointer to
                                                   ///< the batch_matmul
                                                   ///< operation.
    tensorfuncs::adjoint::binary::batch_matmul_fn<Scalar> backward_op =
        tensorfuncs::adjoint::binary::batch_matmul; ///< Function pointer to
                                                    ///< the batch_matmul
                                                    ///< gradient.

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
    Node_batch_matmul(INode *A_ptr, INode *B_ptr, TensorArgs &&...tensor_args)
        : B(B_ptr), INode(A_ptr, tensor_args...) {
        // compute metadata
        Tensor_view A = this->A->value.view();
        Tensor_view B = this->B->value.view();
        Tensor_view C = this->value.view();

        std::vector<int> a_T_shape(A.nDims);
        std::vector<int> a_T_stride(A.nDims);

        std::copy(A.shape, A.shape + A.nDims, a_T_shape.data());
        std::swap(a_T_shape[A.nDims - 1], a_T_shape[A.nDims - 2]);
        std::copy(A.stride, A.stride + A.nDims, a_T_stride.data());
        std::swap(a_T_stride[A.nDims - 1], a_T_stride[A.nDims - 2]);

        Tensor_view a_T = A;
        a_T.shape = a_T_shape.data();
        a_T.stride = a_T_stride.data();

        std::vector<int> b_T_shape(B.nDims);
        std::vector<int> b_T_stride(B.nDims);

        std::copy(B.shape, B.shape + B.nDims, b_T_shape.data());
        std::swap(b_T_shape[B.nDims - 1], b_T_shape[B.nDims - 2]);
        std::copy(B.stride, B.stride + B.nDims, b_T_stride.data());
        std::swap(b_T_stride[B.nDims - 1], b_T_stride[B.nDims - 2]);

        Tensor_view b_T = B;
        b_T.shape = b_T_shape.data();
        b_T.stride = b_T_stride.data();

        detail::batch_matmul_metadata_impl(
            A, B, C, this->strideA[0], this->strideB[0], this->strideC[0],
            this->c_shape[0], this->A_colStride[0], this->B_rowStride[0],
            this->shared_dim[0], this->C_nDims);
        detail::batch_matmul_metadata_impl(
            C, b_T, A, this->strideC[1], this->strideB[1], this->strideA[1],
            this->c_shape[1], this->A_colStride[1], this->B_rowStride[1],
            this->shared_dim[1], this->C_nDims);
        detail::batch_matmul_metadata_impl(
            a_T, C, B, this->strideA[2], this->strideC[2], this->strideB[2],
            this->c_shape[2], this->A_colStride[2], this->B_rowStride[2],
            this->shared_dim[2], this->C_nDims);

        // assign compile-time recursive function
        if (C_nDims <= Dispatchers::MAX_NDIMS) {
            forward_op = Dispatchers::get_batch_matmul<Scalar>()[C_nDims];
            backward_op = Dispatchers::get_batch_matmul_grad<Scalar>()[C_nDims];
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
