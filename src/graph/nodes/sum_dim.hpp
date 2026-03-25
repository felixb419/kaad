#pragma once

#include <cstddef>                      // for size_t
#include <kaad/functions/adjoint.hpp>   // for sum_dim, sum_dim_fn
#include <kaad/functions/primal.hpp>    // for sum_dim, sum_dim_fn
#include <kaad/graph/nodes/inode.hpp>   // for INode
#include <kaad/scalar.hpp>              // for Scalar
#include <kaad/tensor/tensor_types.hpp> // for Stride, ShapeView
#include <vector>                       // for vector

namespace kaad {

/**
 * @brief A sum dim operation node for a @ref kaad::Graph
 * @ingroup nodes
 * @internal
 */
class NodeSumDim : public INode {
  private:
    INode *input = nullptr; ///< Pointer to the input Node.

    functions::primal::unary::sum_dim_fn<Scalar> val_func =
        functions::primal::unary::sum_dim; ///< Function pointer to the
                                           ///< sum_dim operation.
    functions::adjoint::unary::sum_dim_fn<Scalar> grad_func =
        functions::adjoint::unary::sum_dim; ///< Function pointer to the
                                            ///< sum_dim gradient.

    Stride input_stride; ///< stride Array for input.
    Stride value_stride; ///< stride Array for value.
    std::vector<std::size_t>
        input_offset;           ///< Per-dim offset to the end of input buffer.
    std::size_t value_rank = 0; ///< Number of dimensions of value.

    void metadata(int dim);

  public:
    /**
     * @brief Construct sum dim node.
     * @param input_ptr Pointer to the first input node.
     * @param dim Index of the dimension summed over.
     * @param value_shape Output/gradient shape
     */
    NodeSumDim(INode *input_ptr, int dim, ShapeView value_shape);

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
