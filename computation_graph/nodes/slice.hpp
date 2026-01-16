#pragma once

#include "../../tensorfuncs/adjoint_ops.hpp" // for tensorfuncs::adjoint
#include "../../tensorfuncs/primal_ops.hpp"  // for tensorfuncs::primal
#include "../../tensorfuncs/strides.hpp"     // for Strides::slice
#include "dispatchers.hpp"                   // for get_slice, get_slice_grad
#include "inode.hpp"                         // for INode

namespace kaad {

/**
 * @brief A slice operation node in a computation graph.
 * @see tensorfuncs::primal::unary::slice
 * @see tensorfuncs::adjoint::unary::slice
 * @tparam T The scalar type.
 */
template <typename T> struct Node_slice : INode<T> {
    tensorfuncs::primal::unary::slice_fn<T> forward_op =
        tensorfuncs::primal::unary::slice;
    tensorfuncs::adjoint::unary::slice_fn<T> backward_op =
        tensorfuncs::adjoint::unary::slice;

    int *strideA = nullptr;           ///< Stride array for tensor A.
    int *strideB = nullptr;           ///< Stride array for tensor B.
    int *strideC = nullptr;           ///< Stride array for tensor C.
    size_t *start_offset_a = nullptr; ///< Offset for the start of A.
    size_t *C_offset = nullptr; ///< Per-dim offset to the end of C buffer.
    size_t C_nDims = 0;         ///< Number of the dimensions of the C tensor.

    /**
     * @brief Constructs a slice node.
     * @param A_ptr    Pointer to the input node.
     * @param tensor_args       Arguments to construct the output tensor.
     */
    template <typename... TensorArgs>
    Node_slice(INode<T> *A_ptr, const int *offset_arr,
               TensorArgs &&...tensor_args)
        : INode<T>(A_ptr, tensor_args...) {
        Strides::slice(*this, offset_arr);

        size_t a_ndims = static_cast<INode<T> *>(this)->A->value.nDims();
        if (a_ndims < Dispatchers::MAX_NDIMS) {
            forward_op = Dispatchers::get_slice<T>()[a_ndims];
            backward_op = Dispatchers::get_slice_grad<T>()[a_ndims];
        }
    }

    /**
     * @brief Destructor for Node_slice.
     */
    ~Node_slice() {
        delete[] strideA;
        delete[] strideB;
        delete[] strideC;
        delete[] C_offset;
    }

    /**
     * @brief Evaluates the slice operation by applying forward_op, if not
     * already evaluated.
     */
    inline void eval() override {
        if (!this->evaluated) {
            this->A->eval();

            forward_op(this->A->value.data(), this->value.elements_.data(),
                       strideA, strideC, start_offset_a, C_offset, C_nDims);
            this->evaluated = true;
        }
    }

    /**
     * @brief Propagates gradients back through the slice operation by applying
     * backward_op.
     */
    inline void getGrad() override {
        backward_op(this->A->gradient.elements_.data(), this->gradient.data(),
                    strideA, strideC, start_offset_a, C_offset, C_nDims);

        if (this->A->hasInputs) {
            this->A->getGrad();
        }
    }
};

} // namespace kaad
