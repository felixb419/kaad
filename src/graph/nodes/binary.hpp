#pragma once

#include <kaad/functions/adjoint.hpp> // for pointwise_fn
#include <kaad/functions/primal.hpp>  // for pointwise, pointwis...
#include <kaad/graph/nodes/inode.hpp> // for INode
#include <kaad/scalar.hpp>            // for Scalar
#include <kaad/tensor/tensor.hpp>     // for Tensor
#include <span>                       // for span

namespace kaad {

/**
 * @brief A binary operation node for a @ref kaad::Graph
 * @ingroup nodes
 * @internal
 * @tparam Kernel A kernel struct providing `Op` and `Grad` types for the
 * operation.
 */
template <class Kernel> class NodeBinary : public INode {
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
     * @brief Constructs a binary operation node.
     * @ingroup nodes
     * @param operation Function pointer to the value operation.
     * @param derivative Function pointer to the gradient operation.
     * @param lhs_ptr Pointer to the first input node.
     * @param rhs_ptr Pointer to the second input node.
     * @param value_shape Output/gradient shape
     */
    NodeBinary(functions::primal::binary::pointwise_fn<Kernel> operation,
               functions::adjoint::binary::pointwise_fn<Kernel> derivative,
               INode *lhs_ptr, INode *rhs_ptr, ShapeView value_shape)
        : INode(value_shape, false), forward_op(operation),
          backward_op(derivative), lhs(lhs_ptr), rhs(rhs_ptr) {
        auto *base_ptr = static_cast<INode *>(this);
        this->end = base_ptr->value().data() + base_ptr->value().size();
    }

    /// @return Type of the node as a string.
    [[nodiscard]] const char *node_type() const noexcept override {
        return "NodeBinary";
    }

    /// Compute @c value for this node.
    /// Computes @c value for @c lhs and @c rhs first.
    void eval() override {
        if (!this->evaluated()) {
            this->lhs->eval();
            this->rhs->eval();

            forward_op(this->lhs->value().data(), this->rhs->value().data(),
                       this->value().data(), end);
            this->evaluated_ = true;
        }
    }

    /// Compute @c gradient for this node.
    /// Computes @c gradient for @c lhs and @c rhs after.
    void get_grad() override {
        backward_op(this->lhs->value().data(), this->lhs->gradient().data(),
                    this->rhs->value().data(), this->rhs->gradient().data(),
                    this->value().data(), this->gradient().data(), end);

        if (!this->lhs->is_input()) {
            this->lhs->get_grad();
        }
        if (!this->rhs->is_input()) {
            this->rhs->get_grad();
        }
    }
};

} // namespace kaad
