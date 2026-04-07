#pragma once

#include <kaad/functions/flexible.hpp>  // for Flexible
#include <kaad/functions/kernels.hpp>   // for Mul
#include <kaad/graph/nodes/inode.hpp>   // for INode
#include <kaad/scalar.hpp>              // for Scalar
#include <kaad/tensor/tensor_types.hpp> // for ShapeView

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

    functions::Flexible::primal_fn<Kernel> forward_op;

    functions::Flexible::adjoint_fn<Kernel> backward_op;

    functions::Flexible::Metadata mdata;

    void metadata();

  public:
    /**
     * @brief Construct outer product node.
     * @param lhs_ptr Pointer to the first input node.
     * @param rhs_ptr Pointer to the second input node.
     * @param value_shape Output/gradient shape
     */
    NodeOuter(INode *lhs_ptr, INode *rhs_ptr, ShapeView value_shape);

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
