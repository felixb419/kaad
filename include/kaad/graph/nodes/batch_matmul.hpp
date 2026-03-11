#pragma once

#include "../../functions/adjoint.hpp" // for batch_matmul, batch_matmul_fn
#include "../../functions/primal.hpp"  // for batch_matmul, batch_matmul_fn
#include "../../scalar.hpp"            // for Scalar
#include "inode.hpp"                   // for INode
#include <array>                       // for array
#include <cstddef>                     // for size_t
#include <span>                        // for span

namespace kaad {

/**
 * @brief A batch_matmul operation node in a computation graph.
 * @ingroup nodes
 * @see functions::primal::binary::batch_matmul
 * @see functions::adjoint::binary::batch_matmul
 */
class Node_batch_matmul : public INode {
  private:
    INode *lhs = nullptr; ///< Pointer to the second input Node.
    INode *rhs = nullptr; ///< Pointer to the first input Node.

    functions::primal::binary::batch_matmul_fn<Scalar> forward_op =
        functions::primal::binary::batch_matmul; ///< Function pointer to
                                                 ///< the batch_matmul
                                                 ///< operation.
    functions::adjoint::binary::batch_matmul_fn<Scalar> backward_op =
        functions::adjoint::binary::batch_matmul; ///< Function pointer to
                                                  ///< the batch_matmul
                                                  ///< gradient.

    /**
     * @brief Stride arrays for A, B, and C for each stage of computation.
     * @ingroup nodes
     * Index convention:
     * - [0] Forward pass (C = A * B)
     * - [1] Gradient w.r.t. A (dA = dC * B^t)
     * - [2] Gradient w.r.t. B (dB = A^t * dC)
     */
    std::array<int *, 3> lhs_stride;            ///< Stride array for tensor A.
    std::array<int *, 3> rhs_stride;            ///< Stride array for tensor B.
    std::array<int *, 3> value_stride;          ///< Stride array for tensor C.
    std::array<int *, 3> value_shape_broadcast; ///< shape of C (without summing
                                                ///< over batch dimensions).
    std::array<int, 3> lhs_colStride; ///< Gap between columns of the A matrix.
    std::array<int, 3> rhs_rowStride; ///< Gap between rows of the B matrix.
    std::array<int, 3> shared_dim; ///< Shared inner dimension of the tensors.
    std::size_t value_rank =
        0; ///< Number of the dimensions of the value tensor.

    void metadata();

  public:
    /**
     * @brief Returns the type of the node as a string.
     * @ingroup nodes
     */
    [[nodiscard]] const char *node_type() const noexcept override;

    /**
     * @brief Constructs a batch_matmul node.
     * @ingroup nodes
     * @param A_ptr Pointer to the first input node.
     * @param B_ptr Pointer to the second input node.
     * @param value_shape Shape of the value and gradient tensors.
     */
    Node_batch_matmul(INode *lhs_ptr, INode *rhs_ptr,
                      std::span<const int> value_shape);

    /**
     * @brief Destructor for Node_batch_matmul.
     * @ingroup nodes
     */
    ~Node_batch_matmul() noexcept;

    /**
     * @brief Evaluates the batch_matmul operation by calling forward_op, if not
     * @ingroup nodes
     * already evaluated.
     */
    void eval() override;

    /**
     * @brief Propagates gradients back through batch batch_matmul operation by
     * @ingroup nodes
     * calling backward_func.
     */
    void getGrad() override;
};

} // namespace kaad
