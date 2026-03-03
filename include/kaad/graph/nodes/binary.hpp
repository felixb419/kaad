#pragma once

#include "../../functions/adjoint.hpp" // for pointwise_fn
#include "../../functions/primal.hpp"  // for pointwise, pointwis...
#include "../../scalar.hpp"            // for Scalar
#include "../../tensor/tensor.hpp"     // for Tensor
#include "inode.hpp"                   // for INode
#include <span>                        // for span

namespace kaad {

/**
 * @brief A binary operation node in a computation graph.
 * @ingroup nodes
 * @see functions::primal::binary::pointwise
 * @see functions::adjoint::binary::pointwise
 * @tparam Kernel A kernel struct providing `Op` and `Grad` types for the
 * operation.
 */
template <class Kernel> class Node_binary : public INode {
  private:
    functions::primal::binary::pointwise_fn<Kernel> forward_op =
        functions::primal::binary::pointwise<Kernel>; ///< Function pointer to
                                                      ///< the value operation.

    functions::adjoint::binary::pointwise_fn<Kernel> backward_op =
        functions::primal::binary::pointwise<
            Kernel>; ///< Function pointer to the gradient operation.

    INode *lhs = nullptr; ///< Pointer to the first input Node.
    INode *rhs = nullptr; ///< Pointer to the second input Node.

    const Scalar *end =
        nullptr; ///< Pointer to the end of longest buffer (used for iteration,
                 ///< buffer may differ depending on operation).

  public:
    /**
     * @brief Constructs a binary operation node with the given operation and
     * @ingroup nodes
     * gradient.
     * @param operation Function pointer to the value operation.
     * @param derivative Function pointer to the gradient operation.
     * @param lhs_ptr Pointer to the first input node.
     * @param rhs_ptr Pointer to the second input node.
     * @param value_shape Shape of the value and gradient tensors.
     */
    Node_binary(functions::primal::binary::pointwise_fn<Kernel> operation,
                functions::adjoint::binary::pointwise_fn<Kernel> derivative,
                INode *lhs_ptr, INode *rhs_ptr,
                std::span<const int> value_shape)
        : INode(value_shape, false), forward_op(operation),
          backward_op(derivative), lhs(lhs_ptr), rhs(rhs_ptr) {
        INode *base_ptr = static_cast<INode *>(this);
        this->end = base_ptr->value().data() + base_ptr->value().size();
    }

    /**
     * @brief Returns the type of the node as a string.
     * @ingroup nodes
     */
    const char *node_type() const noexcept override { return "Node_binary"; }

    /**
     * @brief Evaluates the binary operation by applying forward_op, if not
     * @ingroup nodes
     * already evaluated.
     */
    inline void eval() override {
        if (!this->evaluated()) {
            this->lhs->eval();
            this->rhs->eval();

            forward_op(this->lhs->value().data(), this->rhs->value().data(),
                       this->value().data(), end);
            this->evaluated_ = true;
        }
    }

    /**
     * @brief Propagates gradients back through the binary operation by applying
     * @ingroup nodes
     * backward_op.
     */
    inline void getGrad() override {
        backward_op(this->lhs->value().data(), this->lhs->gradient().data(),
                    this->rhs->value().data(), this->rhs->gradient().data(),
                    this->value().data(), this->gradient().data(), end);

        if (this->lhs->hasInputs()) {
            this->lhs->getGrad();
        }
        if (this->rhs->hasInputs()) {
            this->rhs->getGrad();
        }
    }
};

} // namespace kaad
