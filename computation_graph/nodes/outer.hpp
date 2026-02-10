#pragma once

#include "../../tensorfuncs/adjoint_ops.hpp" // for tensorfuncs::adjoint
#include "../../tensorfuncs/primal_ops.hpp"  // for tensorfuncs::primal
#include "inode.hpp"                         // for INode, Node_ptr
#include <vector>                            // for std::vector

namespace kaad {

/**
 * @brief A outer prodcut operation node in a computation graph.
 * @see tensorfuncs::primal::binary::flexible
 * @see tensorfuncs::adjoint::binary::flexible
 * @tparam Kernel A kernel struct providing `Op` and `Grad` types for the
 * operation.
 */
class Node_outer : public INode {
  private:
    void metadata();

  public:
    const char *node_type() const noexcept override;

    using Kernel = typename Kernels::Mul<Scalar>;

    INode *B = nullptr; ///< Pointer to the second input Node.

    tensorfuncs::primal::binary::flexible_fn<Scalar, Kernel> forward_op =
        tensorfuncs::primal::binary::flexible<
            Scalar, Kernel>; ///< Function pointer to the value operation.
    tensorfuncs::adjoint::binary::flexible_fn<Scalar, Kernel> backward_op =
        tensorfuncs::adjoint::binary::flexible<
            Scalar,
            Kernel>; ///< Function pointer to the gradient operation.

    std::vector<int> strideA;     ///< stride Array for A.
    std::vector<int> strideB;     ///< stride Array for B.
    std::vector<int> strideC;     ///< stride Array for C.
    std::vector<size_t> C_offset; ///< Per-dim offset to the end of C buffer.
    size_t C_rank = 0;           ///< Number of the dimensions of the C tensor.

    /**
     * @brief Constructs a outer prodcut operation node with outer prodcut
     * operation and gradient.
     * @param A_ptr Pointer to the first input node.
     * @param B_ptr Pointer to the second input node.
     * @param tensor_args Arguments to construct the output tensor.
     */
    template <typename... TensorArgs>
    Node_outer(INode *A_ptr, INode *B_ptr, TensorArgs &&...tensor_args)
        : B(B_ptr), INode(A_ptr, tensor_args...) {

        this->metadata();
    }

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
