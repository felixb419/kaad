#pragma once

#include "../../tensorfuncs/adjoint_ops.hpp" // for tensorfuncs::adjoint
#include "../../tensorfuncs/primal_ops.hpp"  // for tensorfuncs::primal
#include "../../tensorfuncs/strides.hpp"     // for Strides::sum_dim
#include "dispatchers.hpp"                   // for get_sumDim, get_sumDim_grad
#include "inode.hpp"                         // for INode

namespace kaad {

/**
 * @brief A sum_dim operation node in a computation graph.
 * @see tensorfuncs::primal::unary::sum_dim
 * @see tensorfuncs::adjoint::unary::sum_dim
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
    size_t C_nDims = 0;         ///< Number of dimensions of A

    /**
     * @brief Constructs a sum_dim node.
     * @param A_ptr    Pointer to the input node.
     * @param tensor_args       Arguments to construct the output tensor.
     */
    template <typename... TensorArgs>
    Node_sum_dim(INode<T> *A_ptr, int dim, TensorArgs &&...tensor_args)
        : INode<T>(A_ptr, tensor_args...) {
        Strides::sum_dim(*this, dim);

        size_t a_ndims = static_cast<INode<T> *>(this)->A->value.nDims();
        if (a_ndims <= Dispatchers::MAX_NDIMS) {
            val_func = Dispatchers::get_sumDim<T>()[a_ndims];
            grad_func = Dispatchers::get_sumDim_grad<T>()[a_ndims];
        }
    }

    /**
     * @brief Destructor for Node_sum_dim.
     */
    ~Node_sum_dim() {
        delete[] strideA;
        delete[] strideC;
        delete[] A_offset;
    }

    /**
     * @brief Evaluates the sum_dim operation by applying forward_op, if not
     * already evaluated.
     */
    inline void eval() override {
        if (!this->evaluated) {
            this->A->eval();

            val_func(this->A->value.data(), this->value.elements_.data(),
                     strideA, strideC, A_offset, C_nDims);
            this->evaluated = true;
        }
    }

    /**
     * @brief Propagates gradients back through the sum_dim operation, by
     * applying backward_op.
     */
    inline void getGrad() override {
        grad_func(this->A->gradient.elements_.data(), this->gradient.data(),
                  strideA, strideC, A_offset, C_nDims);

        if (this->A->hasInputs) {
            this->A->getGrad();
        }
    }
};

} // namespace kaad
