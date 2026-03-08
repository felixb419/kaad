#pragma once

#include "../../functions/adjoint.hpp" // for dot, dot_fn
#include "../../functions/primal.hpp"  // for dot, dot_fn
#include "../../scalar.hpp"            // for Scalar
#include "inode.hpp"                   // for INode

namespace kaad {

// forward declarations for friend declaration
class Graph;
class Node;

/**
 * @brief A binary operation node in a computation graph.
 * @ingroup nodes
 * @see functions::primal::binary::pointwise
 * @see functions::adjoint::binary::pointwise
 */
class Node_dot : public INode {
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
     * @brief Constructs a binary operation node with the given operation and
     * @ingroup nodes
     * gradient.
     * @param operation Function pointer to the value operation.
     * @param derivative Function pointer to the gradient operation.
     * @param A_ptr Pointer to the first input node.
     * @param B_ptr Pointer to the second input node.
     * @param tensor_args Arguments to construct the output tensor.
     */
    Node_dot(INode *A_ptr, INode *B_ptr);

    /**
     * @brief Returns the type of the node as a string.
     * @ingroup nodes
     */
    const char *node_type() const noexcept override;

    /**
     * @brief Evaluates the binary operation by applying forward_op, if not
     * @ingroup nodes
     * already evaluated.
     */
    void eval() override;

    /**
     * @brief Propagates gradients back through the binary operation by applying
     * @ingroup nodes
     * backward_op.
     */
    void getGrad() override;

    friend Node dot(Graph &rec, Node A, Node B);
};

} // namespace kaad
