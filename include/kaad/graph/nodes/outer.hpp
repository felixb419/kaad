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
 * @brief A outer prodcut operation node in a computation graph.
 * @ingroup nodes
 * @see functions::primal::binary::flexible
 * @see functions::adjoint::binary::flexible
 */
class Node_outer : public INode {
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
     * @brief Constructs a outer prodcut operation node with outer prodcut
     * @ingroup nodes
     * operation and gradient.
     * @param lhs_ptr Pointer to the first input node.
     * @param rhs_ptr Pointer to the second input node.
     * @param value_shape Shape of the value and gradient tensors.
     */
    Node_outer(INode *lhs_ptr, INode *rhs_ptr,
               std::span<const int> value_shape);

    /**
     * @brief Returns the type of the node as a string.
     * @ingroup nodes
     */
    [[nodiscard]] const char *node_type() const noexcept override;

    /**
     * @brief Evaluates the outer prodcut operation by calling forward_op, if
     * @ingroup nodes
     * not already evaluated.
     */
    void eval() override;

    /**
     * @brief Propagates gradients back through the outer prodcut operation, by
     * @ingroup nodes
     * calling backward_op.
     */
    void getGrad() override;
};

} // namespace kaad
