#pragma once

#include <cstddef>                      // for size_t
#include <kaad/functions/adjoint.hpp>   // for mean_dim, mean_dim_fn
#include <kaad/functions/primal.hpp>    // for mean_dim, mean_dim_fn
#include <kaad/graph/nodes/inode.hpp>   // for INode
#include <kaad/scalar.hpp>              // for Scalar
#include <kaad/static_vector.hpp>       // for StaticVector
#include <kaad/tensor/tensor_types.hpp> // for Strides, ShapeView

namespace kaad {

/**
 * @brief A mean dim operation node for a @ref kaad::Graph
 * @ingroup nodes
 * @internal
 */
class NodeMeanDim : public INode {
  private:
    INode *input = nullptr; ///< Pointer to the input Node.

    functions::primal::unary::mean_dim_fn<Scalar> forward_op =
        functions::primal::unary::mean_dim;
    functions::adjoint::unary::mean_dim_fn<Scalar> backward_op =
        functions::adjoint::unary::mean_dim;

    Strides input_strides; ///< Stride array for input tensor.
    Strides value_strides; ///< Stride array for value tensor.
    StaticVector<std::size_t>
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
     * @brief Construct mean dim node.
     * @param lhs_ptr Pointer to the first input node.
     * @param rhs_ptr Pointer to the second input node.
     * @param value_shape Output/gradient shape
     */
    NodeMeanDim(INode *input_ptr, int dim, ShapeView value_shape);

    /// @return Type of the node as a string.
    [[nodiscard]] const char *node_type() const noexcept override;

    /// Compute @c value for this node.
    /// Computes @c value for @c lhs and @c rhs first.
    void eval() override;

    /// Compute @c gradient for this node.
    /// Computes @c gradient for @c lhs and @c rhs after.
    void get_grad() override;
};

} // namespace kaad
