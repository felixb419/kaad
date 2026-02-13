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
  private:
    void metadata(int dim);

  public:
    /**
     * @brief Returns the type of the node as a string.
     */
    const char *node_type() const noexcept override;

    tensorfuncs::primal::unary::sum_dim_fn<Scalar> val_func =
        tensorfuncs::primal::unary::sum_dim; ///< Function pointer to the
                                             ///< sum_dim operation.
    tensorfuncs::adjoint::unary::sum_dim_fn<Scalar> grad_func =
        tensorfuncs::adjoint::unary::sum_dim; ///< Function pointer to the
                                              ///< sum_dim gradient.

    std::vector<int> strideA;     ///< stride Array for A.
    std::vector<int> strideC;     ///< stride Array for C.
    std::vector<size_t> A_offset; ///< Per-dim offset to the end of A buffer.
    size_t C_rank = 0;            ///< Number of dimensions of A

    /**
     * @brief Constructs a sum_dim node.
     * @param A_ptr    Pointer to the input node.
     * @param dim Index of the relevant dimension.
     * @param value_shape Shape of the value and gradient tensors.
     */
    Node_sum_dim(INode *A_ptr, int dim, std::span<const int> value_shape)
        : INode(A_ptr, value_shape) {
        this->metadata(dim);
    }

    /**
     * @brief Evaluates the sum_dim operation by applying forward_op, if not
     * already evaluated.
     */
    void eval() override;

    /**
     * @brief Propagates gradients back through the sum_dim operation, by
     * applying backward_op.
     */
    void getGrad() override;
};

} // namespace kaad
