#pragma once

#include "../../tensorfuncs/adjoint_ops.hpp" // for tensorfuncs::adjoint
#include "../../tensorfuncs/primal_ops.hpp"  // for tensorfuncs::primal
#include "../../tensorfuncs/strides.hpp"     // for Strides::mean_dim
#include "dispatchers.hpp" // for get_meanDim, get_meanDim_grad
#include "inode.hpp"       // for INode

namespace kaad {

/**
 * @brief A mean_dim operation node in a computation graph.
 * @see tensorfuncs::primal::unary::mean_dim
 * @see tensorfuncs::adjoint::unary::mean_dim
 * @tparam T The scalar type.
 */
template <typename T> struct Node_mean_dim : INode<T> {
    tensorfuncs::primal::unary::mean_dim_fn<T> forward_op =
        tensorfuncs::primal::unary::mean_dim;
    tensorfuncs::adjoint::unary::mean_dim_fn<T> backward_op =
        tensorfuncs::adjoint::unary::mean_dim;

    int *strideA = nullptr;     ///< stride Array for A.
    int *strideC = nullptr;     ///< stride Array for C.
    size_t *A_offset = nullptr; ///< Per-dim offset to the end of A buffer.
    const T *C_end =
        nullptr; ///< Pointer to the end of the C buffer (used for iteration).
    const T *dA_end =
        nullptr; ///< Pointer to the end of the dA buffer (used for iteration).
    size_t A_nDims = 0; ///< Number of the dimensions of the A tensor.
    T divisor = 0; ///< Divisor to compute the mean of the A tensor (length of A
                   ///< in relevant dimension).

    /**
     * @brief Constructs a mean_dim node with the given operation and gradient.
     * @param A_ptr Pointer to the input node.
     * @param tensor_args Arguments to construct the output tensor.
     */
    template <typename... TensorArgs>
    Node_mean_dim(INode<T> *A_ptr, int dim, TensorArgs &&...tensor_args)
        : INode<T>(A_ptr, tensor_args...) {
        Strides::mean_dim(*this, dim);

        size_t a_ndims = static_cast<INode<T> *>(this)->A->value.nDims();
        if (a_ndims <= Dispatchers::MAX_NDIMS) {
            forward_op = Dispatchers::get_meanDim<T>()[a_ndims];
            backward_op = Dispatchers::get_meanDim_grad<T>()[a_ndims];
        }
    }

    /**
     * @brief Destructor for Node_mean_dim.
     */
    ~Node_mean_dim() {
        delete[] strideA;
        delete[] strideC;
        delete[] A_offset;
    }

    /**
     * @brief Evaluates the mean_dim operation by applying forward_op, if not
     * already evaluated.
     */
    inline void eval() override {
        if (!this->evaluated) {
            this->A->eval();

            forward_op(this->A->value.data(), this->value.elements_.data(),
                       strideA, strideC, A_offset, A_nDims, divisor, C_end);
            this->evaluated = true;
        }
    }

    /**
     * @brief Propagates gradients back through the mean_dim operation by
     * applying backward_op.
     */
    inline void getGrad() override {
        backward_op(this->A->value.data(), this->A->gradient.elements_.data(),
                    this->value.data(), this->gradient.data(), strideA, strideC,
                    A_offset, A_nDims, divisor, dA_end);

        if (this->A->hasInputs) {
            this->A->getGrad();
        }
    }
};

} // namespace kaad
