#pragma once

#include "../../tensorfuncs/adjoint_ops.hpp" // for tensorfuncs::adjoint
#include "../../tensorfuncs/kernels.hpp"
#include "../../tensorfuncs/primal_ops.hpp" // for tensorfuncs::primal
#include "inode.hpp"                        // for INode, Node_ptr
#include <vector>                           // for std::vector

namespace kaad {

/**
 * @brief A outer prodcut operation node in a computation graph.
 * @see tensorfuncs::primal::binary::flexible
 * @see tensorfuncs::adjoint::binary::flexible
 */
class Node_outer : public INode {
  public:
    /**
     * @brief Returns the type of the node as a string.
     */
    const char *node_type() const noexcept override;

    INode *lhs = nullptr; ///< Pointer to the first input Node.
    INode *rhs = nullptr; ///< Pointer to the second input Node.

    using Kernel = typename Kernels::Mul<Scalar>;

    tensorfuncs::primal::binary::flexible_fn<Kernel> forward_op =
        tensorfuncs::primal::binary::flexible<Kernel>; ///< Function pointer to
                                                       ///< the value operation.

    tensorfuncs::adjoint::binary::flexible_fn<Kernel> backward_op =
        tensorfuncs::adjoint::binary::flexible<
            Kernel>; ///< Function pointer to the gradient operation.

    std::vector<int> lhs_stride;  ///< stride Array for lhs.
    std::vector<int> rhs_stride;  ///< stride Array for rhs.
    std::vector<int> strideC;     ///< stride Array for C.
    std::vector<std::size_t> C_offset; ///< Per-dim offset to the end of C buffer.
    std::size_t C_rank = 0;            ///< Number of the dimensions of the C tensor.

    /**
     * @brief Constructs a outer prodcut operation node with outer prodcut
     * operation and gradient.
     * @param lhs_ptr Pointer to the first input node.
     * @param rhs_ptr Pointer to the second input node.
     * @param value_shape Shape of the value and gradient tensors.
     */
    Node_outer(INode *lhs_ptr, INode *rhs_ptr,
               std::span<const int> value_shape);

    /**
     * @brief Evaluates the outer prodcut operation by calling forward_op, if
     * not already evaluated.
     */
    void eval() override;

    /**
     * @brief Propagates gradients back through the outer prodcut operation, by
     * calling backward_op.
     */
    void getGrad() override;
};

} // namespace kaad
