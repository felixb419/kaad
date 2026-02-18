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
    /**
     * @brief Returns the type of the node as a string.
     */
    const char *node_type() const noexcept override { return "Node_binary"; }

    INode *lhs = nullptr; ///< Pointer to the first input Node.
    INode *rhs = nullptr; ///< Pointer to the second input Node.

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
     * @param lhs_ptr Pointer to the first input node.
     * @param rhs_ptr Pointer to the second input node.
     * @param value_shape Shape of the value and gradient tensors.
     */
    Node_binary(tensorfuncs::primal::binary::pointwise_fn<Kernel> operation,
                tensorfuncs::adjoint::binary::pointwise_fn<Kernel> derivative,
                INode *lhs_ptr, INode *rhs_ptr,
                std::span<const int> value_shape)
        : lhs(lhs_ptr), rhs(rhs_ptr), forward_op(operation),
          backward_op(derivative), INode(value_shape, false) {
        INode *base_ptr = static_cast<INode *>(this);
        this->end = base_ptr->value.data() + base_ptr->value.size();
    }

    /**
     * @brief Evaluates the binary operation by applying forward_op, if not
     * already evaluated.
     */
    inline void eval() override {
        if (!this->evaluated) {
            this->lhs->eval();
            this->rhs->eval();

            forward_op(this->lhs->value.data(), this->rhs->value.data(),
                       this->value.elements_.data(), end);
            this->evaluated = true;
        }
    }

    /**
     * @brief Propagates gradients back through the binary operation by applying
     * backward_op.
     */
    inline void getGrad() override {
        backward_op(
            this->lhs->value.data(), this->lhs->gradient.elements_.data(),
            this->rhs->value.data(), this->rhs->gradient.elements_.data(),
            this->value.data(), this->gradient.data(), end);

        if (this->lhs->hasInputs) {
            this->lhs->getGrad();
        }
        if (this->rhs->hasInputs) {
            this->rhs->getGrad();
        }
    }
};

} // namespace kaad
