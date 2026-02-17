#pragma once

#include "../../tensorfuncs/adjoint_ops.hpp" // for tensorfuncs::adjoint
#include "../../tensorfuncs/primal_ops.hpp"  // for tensorfuncs::primal
#include "../dispatchers.hpp"                // for get_flexOp, get_flexGrad
#include "inode.hpp"                         // for INode, Node_ptr
#include <vector>                            // for std::vector

namespace kaad {

template <class Kernel> class Node_binary_flex;

template <class Kernel>
void Node_binary_flex_metadata(Node_binary_flex<Kernel> &node) {
    // compute metadata
    Tensor &A = node.A->value;
    Tensor &B = node.B->value;
    Tensor &C = node.value;

    node.C_rank = C.rank();
    node.strideA.resize(node.C_rank);
    node.strideB.resize(node.C_rank);
    node.strideC.resize(node.C_rank);

    int idx, idxA, idxB, idxC;
    for (int i = 1; i <= node.C_rank; i++) {
        idx = node.C_rank - i;
        idxA = A.rank() - i;
        node.strideA[idx] = idxA >= 0 ? A.stride()[idxA] : 0;
        idxB = B.rank() - i;
        node.strideB[idx] = idxB >= 0 ? B.stride()[idxB] : 0;
        idxC = C.rank() - i;
        node.strideC[idx] = idxC >= 0 ? C.stride()[idxC] : 0;
        // make sure strideC[idx] is 1 instead of 0 if C.shape[idx] is 1 for
        // traversing in flexible function
        if (node.strideC[idx] == 0 && C.shape()[idxC] == 1) {
            node.strideC[idx] = 1;
        }
    }

    node.C_offset.resize(node.C_rank);
    for (int i = 0; i < node.C_rank; i++) {
        node.C_offset[i] = C.shape()[i] * node.strideC[i];
    }

    // assign compile-time recursive function
    if (node.C_rank <= Dispatchers::MAX_NDIMS) {
        node.forward_op = Dispatchers::get_flexOp<Kernel>()[node.C_rank];
        node.backward_op = Dispatchers::get_flexGrad<Kernel>()[node.C_rank];
    }
}

/**
 * @brief A binary_flex operation node in a computation graph.
 * @see tensorfuncs::primal::binary::flexible
 * @see tensorfuncs::adjoint::binary::flexible
 * @tparam Kernel A kernel struct providing `Op` and `Grad` types for the
 * operation.
 */
template <class Kernel> class Node_binary_flex : public INode {
  public:
    /**
     * @brief Returns the type of the node as a string.
     */
    const char *node_type() const noexcept override {
        return "Node_binary_flex";
    }

    INode *A = nullptr; ///< Pointer to the first input Node.
    INode *B = nullptr; ///< Pointer to the second input Node.

    tensorfuncs::primal::binary::flexible_fn<Kernel> forward_op =
        tensorfuncs::primal::binary::flexible<Kernel>; ///< Function pointer to
                                                       ///< the value operation.

    tensorfuncs::adjoint::binary::flexible_fn<Kernel> backward_op =
        tensorfuncs::adjoint::binary::flexible<
            Kernel>; ///< Function pointer to the gradient operation.

    std::vector<int> strideA;     ///< stride Array for A.
    std::vector<int> strideB;     ///< stride Array for B.
    std::vector<int> strideC;     ///< stride Array for C.
    std::vector<size_t> C_offset; ///< Per-dim offset to the end of C buffer.
    size_t C_rank = 0;            ///< Number of the dimensions of the C tensor.

    /**
     * @brief Constructs a binary_flex operation node with binary_flex operation
     * and gradient.
     * @param A_ptr Pointer to the first input node.
     * @param B_ptr Pointer to the second input node.
     * @param value_shape Shape of the value and gradient tensors.
     */
    Node_binary_flex(INode *A_ptr, INode *B_ptr,
                     std::span<const int> value_shape)
        : A(A_ptr), B(B_ptr), INode(value_shape, false) {

        Node_binary_flex_metadata(*this);
    }

    /**
     * @brief Evaluates the binary_flex operation by calling forward_op, if not
     * already evaluated.
     */
    inline void eval() override {
        if (!this->evaluated) {
            this->A->eval();
            this->B->eval();

            forward_op(this->A->value.data(), this->B->value.data(),
                       this->value.elements_.data(), strideA.data(),
                       strideB.data(), strideC.data(), C_offset.data(), C_rank);
            this->evaluated = true;
        }
    }

    /**
     * @brief Propagates gradients back through the binary_flex operation, by
     * calling backward_op.
     */
    inline void getGrad() override {
        backward_op(this->A->value.data(), this->A->gradient.elements_.data(),
                    this->B->value.data(), this->B->gradient.elements_.data(),
                    this->value.data(), this->gradient.data(), strideA.data(),
                    strideB.data(), strideC.data(), C_offset.data(), C_rank);

        if (this->A->hasInputs) {
            this->A->getGrad();
        }
        if (this->B->hasInputs) {
            this->B->getGrad();
        }
    }
};

} // namespace kaad
