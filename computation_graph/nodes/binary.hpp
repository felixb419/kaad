#pragma once

#include "../../tensorfuncs/adjoint_ops.hpp" // for tensorfuncs::adjoint
#include "../../tensorfuncs/primal_ops.hpp"  // for tensorfuncs::primal
#include "inode.hpp"                         // for INode

namespace kaad {

/**
 * @brief A binary operation node in a computation graph.
 * @see tensorfuncs::primal::binary::pointwise
 * @see tensorfuncs::adjoint::binary::pointwise
 * @tparam Kernel A kernel struct providing `Op` and `Grad` types for the
 * operation.
 */
template <class Kernel> class Node_binary : public INode {
  public:
    const char *node_type() const noexcept override { return "Node_binary"; }

    INode *B = nullptr; ///< Pointer to the second input Node.

    tensorfuncs::primal::binary::pointwise_fn<Kernel> forward_op =
        tensorfuncs::primal::binary::pointwise<
            Kernel>; ///< Function pointer to the value operation.

    tensorfuncs::adjoint::binary::pointwise_fn<Kernel> backward_op =
        tensorfuncs::primal::binary::pointwise<
            Kernel>; ///< Function pointer to the gradient operation.

    const Scalar *end =
        nullptr; ///< Pointer to the end of longest buffer (used for iteration,
                 ///< buffer may differ depending on operation).

    /**
     * @brief Constructs a binary operation node with the given operation and
     * gradient.
     * @param operation Function pointer to the value operation.
     * @param derivative Function pointer to the gradient operation.
     * @param A_ptr Pointer to the first input node.
     * @param B_ptr Pointer to the second input node.
     * @param value_shape Shape of the value and gradient tensors.
     */
    Node_binary(tensorfuncs::primal::binary::pointwise_fn<Kernel> operation,
                tensorfuncs::adjoint::binary::pointwise_fn<Kernel> derivative,
                INode *A_ptr, INode *B_ptr, std::span<const int> value_shape)
        : B(B_ptr), forward_op(operation), backward_op(derivative),
          INode(A_ptr, value_shape) {
        INode *base_ptr = static_cast<INode *>(this);
        this->end = base_ptr->value.data() + base_ptr->value.size();
    }

    /**
     * @brief Evaluates the binary operation by applying forward_op, if not
     * already evaluated.
     */
    inline void eval() override {
        if (!this->evaluated) {
            this->A->eval();
            this->B->eval();

            forward_op(this->A->value.data(), this->B->value.data(),
                       this->value.elements_.data(), end);
            this->evaluated = true;
        }
    }

    /**
     * @brief Propagates gradients back through the binary operation by applying
     * backward_op.
     */
    inline void getGrad() override {
        backward_op(this->A->value.data(), this->A->gradient.elements_.data(),
                    this->B->value.data(), this->B->gradient.elements_.data(),
                    this->value.data(), this->gradient.data(), end);

        if (this->A->hasInputs) {
            this->A->getGrad();
        }
        if (this->B->hasInputs) {
            this->B->getGrad();
        }
    }
};

} // namespace kaad
