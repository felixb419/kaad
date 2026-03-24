#pragma once

#include <cstddef>                    // for size_t
#include <kaad/functions/adjoint.hpp> // for flexible, flexible_fn
#include <kaad/functions/kernels.hpp> // for Mul
#include <kaad/functions/primal.hpp>  // for flexible, flexible_fn
#include <kaad/graph/nodes/inode.hpp> // for INode
#include <kaad/scalar.hpp>            // for Scalar
#include <span>                       // for span
#include <vector>                     // for vector

namespace kaad {

/**
 * @brief A outer operation node for a @ref kaad::Graph
 * @ingroup nodes
 * @internal
 */
class NodeOuter : public INode {
  private:
    INode *lhs = nullptr; ///< Pointer to the first input Node.
    INode *rhs = nullptr; ///< Pointer to the second input Node.

    using Kernel = Kernels::Mul<Scalar>;

    functions::primal::binary::flexible_fn<Kernel> forward_op =
        functions::primal::binary::flexible<Kernel>; ///< Function pointer to
                                                     ///< the value operation.

    functions::adjoint::binary::flexible_fn<Kernel> backward_op =
        functions::adjoint::binary::flexible<
            Kernel>; ///< Function pointer to the gradient operation.

    std::vector<int> lhs_stride; ///< stride Array for lhs.
    std::vector<int> rhs_stride; ///< stride Array for rhs.
    std::vector<int> res_stride; ///< stride Array for res.
    std::vector<std::size_t>
        res_offset;           ///< Per-dim offset to the end of res buffer.
    std::size_t res_rank = 0; ///< Number of the dimensions of the res tensor.

    void metadata();

  public:
    /**
     * @brief Construct outer product node.
     * @param lhs_ptr Pointer to the first input node.
     * @param rhs_ptr Pointer to the second input node.
     * @param value_shape Output/gradient shape
     */
    NodeOuter(INode *lhs_ptr, INode *rhs_ptr, std::span<const int> value_shape);

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
