#pragma once

#include <kaad/functions/dot_product.hpp> // for DotProduct
#include <kaad/graph/nodes/inode.hpp>     // for INode
#include <kaad/scalar.hpp>                // for Scalar

namespace kaad {

// forward declarations for friend declaration
class Graph;
class Node;

/**
 * @brief A dot product node for a @ref kaad::Graph
 * @ingroup nodes
 * @internal
 */
class NodeDot : public INode {
  private:
    INode *lhs = nullptr;
    INode *rhs = nullptr;

    functions::DotProduct::primal_fn forward_op = functions::DotProduct::primal;

    functions::DotProduct::adjoint_fn backward_op =
        functions::DotProduct::adjoint;

    const Scalar *end; ///< Is set to the end of the elements of @c lhs or @c
                       ///< rhs depending on inputs.

  public:
    /**
     * @brief Construct dot product node.
     * @param lhs_ptr Pointer to the first input node.
     * @param rhs_ptr Pointer to the second input node.
     */
    NodeDot(INode *lhs_ptr, INode *rhs_ptr);

    /// @return Type of the node as a string.
    [[nodiscard]] const char *node_type() const noexcept override;

    /// Compute @c value for this node.
    /// Computes @c value for @c lhs and @c rhs first.
    void eval() override;

    /// Compute @c gradient for this node.
    /// Computes @c gradient for @c lhs and @c rhs after.
    void get_grad() override;

    friend Node dot(Graph &rec, Node lhs, Node rhs);
};

} // namespace kaad
