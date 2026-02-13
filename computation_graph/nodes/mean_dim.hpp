#pragma once

#include "../../tensorfuncs/adjoint_ops.hpp" // for tensorfuncs::adjoint
#include "../../tensorfuncs/primal_ops.hpp"  // for tensorfuncs::primal
#include "inode.hpp"                         // for INode

namespace kaad {

/**
 * @brief A mean_dim operation node in a computation graph.
 * @see tensorfuncs::primal::unary::mean_dim
 * @see tensorfuncs::adjoint::unary::mean_dim
 */
class Node_mean_dim : public INode {
  private:
    void metadata(int dim);

  public:
    const char *node_type() const noexcept override;

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
    size_t A_rank = 0;  ///< Number of the dimensions of the A tensor.
    Scalar divisor = 0; ///< Divisor to compute the mean of the A tensor (length
                        ///< of A in relevant dimension).

    /**
     * @brief Constructs a mean_dim node with the given operation and gradient.
     * @param A_ptr Pointer to the input node.
     * @param dim Index of the relevant dimension.
     * @param value_shape Shape of the value and gradient tensors.
     */
    Node_mean_dim(INode *A_ptr, int dim, std::span<const int> value_shape)
        : INode(A_ptr, value_shape) {

        this->metadata(dim);
    }

    /**
     * @brief Evaluates the mean_dim operation by applying forward_op, if not
     * already evaluated.
     */
    void eval() override;

    /**
     * @brief Propagates gradients back through the mean_dim operation by
     * applying backward_op.
     */
    void getGrad() override;
};

} // namespace kaad
