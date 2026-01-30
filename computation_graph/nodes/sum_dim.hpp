#pragma once

#include "../../tensorfuncs/adjoint_ops.hpp" // for tensorfuncs::adjoint
#include "../../tensorfuncs/primal_ops.hpp"  // for tensorfuncs::primal
#include "../common.hpp"                     // for along_dim_metadata_impl
#include "../dispatchers.hpp"                // for get_sumDim, get_sumDim_grad
#include "inode.hpp"                         // for INode

namespace kaad {

/**
 * @brief A sum_dim operation node in a computation graph.
 * @see tensorfuncs::primal::unary::sum_dim
 * @see tensorfuncs::adjoint::unary::sum_dim
 */
class Node_sum_dim : public INode {
  public:
    const char *node_type() const noexcept override { return "Node_sum_dim"; }

    tensorfuncs::primal::unary::sum_dim_fn<Scalar> val_func =
        tensorfuncs::primal::unary::sum_dim; ///< Function pointer to the
                                             ///< sum_dim operation.
    tensorfuncs::adjoint::unary::sum_dim_fn<Scalar> grad_func =
        tensorfuncs::adjoint::unary::sum_dim; ///< Function pointer to the
                                              ///< sum_dim gradient.

    std::vector<int> strideA;     ///< stride Array for A.
    std::vector<int> strideC;     ///< stride Array for C.
    std::vector<size_t> A_offset; ///< Per-dim offset to the end of A buffer.
    size_t C_nDims = 0;           ///< Number of dimensions of A

    /**
     * @brief Constructs a sum_dim node.
     * @param A_ptr    Pointer to the input node.
     * @param tensor_args       Arguments to construct the output tensor.
     */
    template <typename... TensorArgs>
    Node_sum_dim(INode *A_ptr, int dim, TensorArgs &&...tensor_args)
        : INode(A_ptr, tensor_args...) {
        // compute metadata
        Tensor_view A = this->A->value.view();
        Tensor_view C = this->value.view();

        detail::along_dim_metadata_impl<Scalar>(A, C, dim, this->C_nDims,
                                                this->A_offset, this->strideA,
                                                this->strideC);

        // assign compile-time recursive function
        size_t a_ndims = static_cast<INode *>(this)->A->value.nDims();
        if (a_ndims <= Dispatchers::MAX_NDIMS) {
            val_func = Dispatchers::get_sumDim<Scalar>()[a_ndims];
            grad_func = Dispatchers::get_sumDim_grad<Scalar>()[a_ndims];
        }
    }

    /**
     * @brief Evaluates the sum_dim operation by applying forward_op, if not
     * already evaluated.
     */
    inline void eval() override {
        if (!this->evaluated) {
            this->A->eval();

            val_func(this->A->value.data(), this->value.elements_.data(),
                     strideA.data(), strideC.data(), A_offset.data(), C_nDims);
            this->evaluated = true;
        }
    }

    /**
     * @brief Propagates gradients back through the sum_dim operation, by
     * applying backward_op.
     */
    inline void getGrad() override {
        grad_func(this->A->gradient.elements_.data(), this->gradient.data(),
                  strideA.data(), strideC.data(), A_offset.data(), C_nDims);

        if (this->A->hasInputs) {
            this->A->getGrad();
        }
    }
};

} // namespace kaad
