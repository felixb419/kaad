#pragma once

#include <cstddef>                    // for size_t
#include <kaad/functions/adjoint.hpp> // for sum_dim, sum_dim_fn
#include <kaad/functions/primal.hpp>  // for sum_dim, sum_dim_fn
#include <kaad/graph/nodes/inode.hpp> // for INode
#include <kaad/scalar.hpp>            // for Scalar
#include <span>                       // for span
#include <vector>                     // for vector

namespace kaad {

/**
 * @brief A sum_dim operation node in a computation graph.
 * @ingroup nodes
 * @see functions::primal::unary::sum_dim
 * @see functions::adjoint::unary::sum_dim
 */
class Node_sum_dim : public INode {
  private:
    INode *input = nullptr; ///< Pointer to the input Node.

    functions::primal::unary::sum_dim_fn<Scalar> val_func =
        functions::primal::unary::sum_dim; ///< Function pointer to the
                                           ///< sum_dim operation.
    functions::adjoint::unary::sum_dim_fn<Scalar> grad_func =
        functions::adjoint::unary::sum_dim; ///< Function pointer to the
                                            ///< sum_dim gradient.

    std::vector<int> input_stride; ///< stride Array for input.
    std::vector<int> value_stride; ///< stride Array for value.
    std::vector<std::size_t>
        input_offset;           ///< Per-dim offset to the end of input buffer.
    std::size_t value_rank = 0; ///< Number of dimensions of value.

    void metadata(int dim);

  public:
    /**
     * @brief Constructrs a sum_dim node.
     * @ingroup nodes
     * @param input_ptr    Pointer to the input node.
     * @param dim Index of the relevant dimension.
     * @param value_shape Shape of the value and gradient tensors.
     */
    Node_sum_dim(INode *input_ptr, int dim, std::span<const int> value_shape);

    /**
     * @brief Returns the type of the node as a string.
     * @ingroup nodes
     */
    [[nodiscard]] const char *node_type() const noexcept override;

    /**
     * @brief Evaluates the sum_dim operation by applying forward_op, if not
     * @ingroup nodes
     * already evaluated.
     */
    void eval() override;

    /**
     * @brief Propagates gradients back through the sum_dim operation, by
     * @ingroup nodes
     * applying backward_op.
     */
    void getGrad() override;
};

} // namespace kaad
