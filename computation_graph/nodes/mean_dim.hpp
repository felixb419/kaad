#pragma once

#include "../../tensorfuncs/adjoint_ops.hpp" // for tensorfuncs::adjoint
#include "../../tensorfuncs/primal_ops.hpp"  // for tensorfuncs::primal
#include "../common.hpp"                     // for along_dim_metadata_impl
#include "../dispatchers.hpp" // for get_meanDim, get_meanDim_grad
#include "inode.hpp"          // for INode

namespace kaad {

/**
 * @brief A mean_dim operation node in a computation graph.
 * @see tensorfuncs::primal::unary::mean_dim
 * @see tensorfuncs::adjoint::unary::mean_dim
 */
class Node_mean_dim : public INode {
  public:
    const char *node_type() const noexcept override { return "Node_mean_dim"; }

    tensorfuncs::primal::unary::mean_dim_fn<Scalar> forward_op =
        tensorfuncs::primal::unary::mean_dim;
    tensorfuncs::adjoint::unary::mean_dim_fn<Scalar> backward_op =
        tensorfuncs::adjoint::unary::mean_dim;

    std::vector<int> strideA;     ///< stride Array for A.
    std::vector<int> strideC;     ///< stride Array for C.
    std::vector<size_t> A_offset; ///< Per-dim offset to the end of A buffer.
    const Scalar *C_end =
        nullptr; ///< Pointer to the end of the C buffer (used for iteration).
    const Scalar *dA_end =
        nullptr; ///< Pointer to the end of the dA buffer (used for iteration).
    size_t A_nDims = 0; ///< Number of the dimensions of the A tensor.
    Scalar divisor = 0; ///< Divisor to compute the mean of the A tensor (length
                        ///< of A in relevant dimension).

    /**
     * @brief Constructs a mean_dim node with the given operation and gradient.
     * @param A_ptr Pointer to the input node.
     * @param tensor_args Arguments to construct the output tensor.
     */
    template <typename... TensorArgs>
    Node_mean_dim(INode *A_ptr, int dim, TensorArgs &&...tensor_args)
        : INode(A_ptr, tensor_args...) {
        // compute metadata
        Tensor_view A = this->A->value.view();
        Tensor_view C = this->value.view();
        Tensor_view dA = this->A->gradient.view();

        this->divisor = A.shape[dim];
        this->C_end = C.elements + C.len;
        this->dA_end = dA.elements + dA.len;

        detail::along_dim_metadata_impl(A, C, dim, this->A_nDims,
                                        this->A_offset, this->strideA,
                                        this->strideC);

        // assign compile-time recursive function
        size_t a_ndims = static_cast<INode *>(this)->A->value.nDims();
        if (a_ndims <= Dispatchers::MAX_NDIMS) {
            forward_op = Dispatchers::get_meanDim<Scalar>()[a_ndims];
            backward_op = Dispatchers::get_meanDim_grad<Scalar>()[a_ndims];
        }
    }

    /**
     * @brief Evaluates the mean_dim operation by applying forward_op, if not
     * already evaluated.
     */
    inline void eval() override {
        if (!this->evaluated) {
            this->A->eval();

            forward_op(this->A->value.data(), this->value.elements_.data(),
                       strideA.data(), strideC.data(), A_offset.data(), A_nDims,
                       divisor, C_end);
            this->evaluated = true;
        }
    }

    /**
     * @brief Propagates gradients back through the mean_dim operation by
     * applying backward_op.
     */
    inline void getGrad() override {
        backward_op(this->A->value.data(), this->A->gradient.elements_.data(),
                    this->value.data(), this->gradient.data(), strideA.data(),
                    strideC.data(), A_offset.data(), A_nDims, divisor, dA_end);

        if (this->A->hasInputs) {
            this->A->getGrad();
        }
    }
};

} // namespace kaad
