#pragma once

#include "../../tensorfuncs/adjoint_ops.hpp" // for tensorfuncs::adjoint
#include "../../tensorfuncs/primal_ops.hpp"  // for tensorfuncs::primal
#include "inode.hpp"                         // for INode

namespace kaad {

/**
 * @brief A unary operation node in a computation graph.
 * @see tensorfuncs::primal::unary::pointwise
 * @see tensorfuncs::adjoint::unary::pointwise
 * @tparam Kernel A kernel struct providing `Op` and `Grad` types for the
 * operation.
 */
template <class Kernel> class Node_unary : public INode {
  public:
    /**
     * @brief Returns the type of the node as a string.
     */
    const char *node_type() const noexcept override { return "Node_unary"; }

    INode *input = nullptr; ///< Pointer to the input Node.

    tensorfuncs::primal::unary::pointwise_fn<Kernel> forward_op =
        tensorfuncs::primal::unary::pointwise<Kernel>; ///< Function pointer to
                                                       ///< the value operation.

    tensorfuncs::adjoint::unary::pointwise_fn<Kernel> backward_op =
        tensorfuncs::adjoint::unary::pointwise<
            Kernel>; ///< Function pointer to the gradient operation.

    const Scalar *end =
        nullptr; ///< Pointer to the end of longest buffer (used for iteration,
                 ///< buffer may differ depending on operation).

    /**
     * @brief Constructs a unary node with the given operation and gradient.
     * @param operation  Function pointer to the value operation.
     * @param derivative Function pointer to the gradient operation.
     * @param input_ptr    Pointer to the input node.
     * @param value_shape Shape of the value and gradient tensors.
     */
    Node_unary(tensorfuncs::primal::unary::pointwise_fn<Kernel> operation,
               tensorfuncs::adjoint::unary::pointwise_fn<Kernel> derivative,
               INode *input_ptr, std::span<const int> value_shape)
        : forward_op(operation), backward_op(derivative), input(input_ptr),
          INode(value_shape, false) {
        this->end =
            this->value().data() +
            this->value()
                .size(); // Points to end of value if not overriden in operator.
    }

    /**
     * @brief Evaluates the unary operation by applying forward_op, if not
     * already evaluated.
     */
    inline void eval() override {
        if (!this->evaluated()) {
            this->input->eval();

            forward_op(this->input->value().data(),
                       this->value().elements_.data(), end);
            this->evaluated_ = true;
        }
    }

    /**
     * @brief Propagates gradients back through the unary operation, by applying
     * backward_op.
     */
    inline void getGrad() override {
        backward_op(this->input->value().data(),
                    this->input->gradient().elements_.data(),
                    this->value().data(), this->gradient().data(), end);

        if (this->input->hasInputs()) {
            this->input->getGrad();
        }
    }
};

} // namespace kaad
