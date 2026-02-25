#pragma once

#include "../../functions/adjoint.hpp" // for mean_dim, mean_dim_fn
#include "../../functions/primal.hpp"  // for mean_dim, mean_dim_fn
#include "../../scalar.hpp"            // for Scalar
#include "inode.hpp"                   // for INode
#include <cstddef>                     // for size_t
#include <span>                        // for span
#include <vector>                      // for vector

namespace kaad {

/**
 * @brief A mean_dim operation node in a computation graph.
 * @see functions::primal::unary::mean_dim
 * @see functions::adjoint::unary::mean_dim
 */
class Node_mean_dim : public INode {
  private:
    INode *input = nullptr; ///< Pointer to the input Node.

    functions::primal::unary::mean_dim_fn<Scalar> forward_op =
        functions::primal::unary::mean_dim;
    functions::adjoint::unary::mean_dim_fn<Scalar> backward_op =
        functions::adjoint::unary::mean_dim;

    std::vector<int> input_stride; ///< stride Array for input tensor.
    std::vector<int> value_stride; ///< stride Array for value tensor.
    std::vector<std::size_t>
        input_offset; ///< Per-dim offset to the end of input buffer.
    const Scalar *value_end = nullptr; ///< Pointer to the end of the value
                                       ///< buffer (used for iteration).
    const Scalar *input_grad_end =
        nullptr; ///< Pointer to the end of the value gradient buffer (used for
                 ///< iteration).
    std::size_t input_rank =
        0;              ///< Number of the dimensions of the input tensor.
    Scalar divisor = 0; ///< Divisor to compute the mean of the input tensor
                        ///< (length of A in relevant dimension).

    void metadata(int dim);

  public:
    /**
     * @brief Constructs a mean_dim node with the given operation and gradient.
     * @param lhs_ptr Pointer to the input node.
     * @param dim Index of the relevant dimension.
     * @param value_shape Shape of the value and gradient tensors.
     */
    Node_mean_dim(INode *input_ptr, int dim, std::span<const int> value_shape);

    /**
     * @brief Returns the type of the node as a string.
     */
    const char *node_type() const noexcept override;

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
