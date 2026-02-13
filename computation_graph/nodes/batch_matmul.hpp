#pragma once

#include "../../scalar.hpp"                  // for Scalar
#include "../../tensorfuncs/adjoint_ops.hpp" // for tensorfuncs::adjoint
#include "../../tensorfuncs/primal_ops.hpp"  // for tensorfuncs::primal
#include "inode.hpp"                         // for INode

namespace kaad {

class Node_batch_matmul;

/**
 * @brief A batch_matmul operation node in a computation graph.
 * @see tensorfuncs::primal::binary::batch_matmul
 * @see tensorfuncs::adjoint::binary::batch_matmul
 */
class Node_batch_matmul : public INode {
  private:
    void metadata();

  public:
    /**
     * @brief Returns the type of the node as a string.
     */
    const char *node_type() const noexcept override;

    INode *B = nullptr; ///< Pointer to the second input Node.

    tensorfuncs::primal::binary::batch_matmul_fn<Scalar> forward_op =
        tensorfuncs::primal::binary::batch_matmul; ///< Function pointer to
                                                   ///< the batch_matmul
                                                   ///< operation.
    tensorfuncs::adjoint::binary::batch_matmul_fn<Scalar> backward_op =
        tensorfuncs::adjoint::binary::batch_matmul; ///< Function pointer to
                                                    ///< the batch_matmul
                                                    ///< gradient.

    /**
     * @brief Stride arrays for A, B, and C for each stage of computation.
     * Index convention:
     * - [0] Forward pass (C = A * B)
     * - [1] Gradient w.r.t. A (dA = dC * Bᵗ)
     * - [2] Gradient w.r.t. B (dB = Aᵗ * dC)
     */
    int *(strideA[3]);           ///< Stride array for tensor A.
    int *(strideB[3]);           ///< Stride array for tensor B.
    int *(strideC[3]);           ///< Stride array for tensor C.
    int *(c_shape_broadcast[3]); ///< shape of C (without summing over batch
                                 ///< dimensions).
    int A_colStride[3];          ///< Gap between columns of the A matrix.
    int B_rowStride[3];          ///< Gap between rows of the B matrix.
    int shared_dim[3];           ///< Shared inner dimension of the tensors.
    size_t C_rank = 0; ///< Number of the dimensions of the value tensor.

    /**
     * @brief Constructs a batch_matmul node.
     * @param A_ptr Pointer to the first input node.
     * @param B_ptr Pointer to the second input node.
     * @param value_shape Shape of the value and gradient tensors.
     */
    Node_batch_matmul(INode *A_ptr, INode *B_ptr,
                      std::span<const int> value_shape)
        : B(B_ptr), INode(A_ptr, value_shape) {

        this->metadata();
    }

    /**
     * @brief Destructor for Node_batch_matmul.
     */
    ~Node_batch_matmul() noexcept;

    /**
     * @brief Evaluates the batch_matmul operation by calling forward_op, if not
     * already evaluated.
     */
    void eval() override;

    /**
     * @brief Propagates gradients back through batch batch_matmul operation by
     * calling backward_func.
     */
    void getGrad() override;
};

} // namespace kaad
