#pragma once

#include <kaad/functions/adjoint.hpp> // for dot, dot_fn
#include <kaad/functions/primal.hpp>  // for dot, dot_fn
#include <kaad/graph/nodes/inode.hpp> // for INode
#include <kaad/scalar.hpp>            // for Scalar

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
    INode *lhs = nullptr; ///< Pointer to the first input Node.
    INode *rhs = nullptr; ///< Pointer to the second input Node.

    functions::primal::binary::dot_fn<Scalar> forward_op =
        functions::primal::binary::dot<Scalar>; ///< Function pointer to
                                                ///< the value
                                                ///< operation.

    functions::adjoint::binary::dot_fn<Scalar> backward_op =
        functions::adjoint::binary::dot<Scalar>; ///< Function pointer to the
                                                 ///< gradient operation.

    const Scalar *lhs_end =
        nullptr; ///< Pointer to the end of the A buffer (used for iteration).

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
    void getGrad() override;

    friend Node dot(Graph &rec, Node lhs, Node rhs);
};

} // namespace kaad
