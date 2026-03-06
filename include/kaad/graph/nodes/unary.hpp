#pragma once

#include "../../functions/adjoint.hpp" // for pointwise_fn, pointwise
#include "../../functions/primal.hpp"  // for pointwise_fn, pointwise
#include "../../scalar.hpp"            // for Scalar
#include "inode.hpp"                   // for INode
#include <span>                        // for span

namespace kaad {

// forward declarations for friend declaration
class Computation_graph;
class Node;

/**
 * @brief A unary operation node in a computation graph.
 * @ingroup nodes
 * @see functions::primal::unary::pointwise
 * @see functions::adjoint::unary::pointwise
 * @tparam Kernel A kernel struct providing `Op` and `Grad` types for the
 * operation.
 */
template <class Kernel> class Node_unary : public INode {
  private:
    functions::primal::unary::pointwise_fn<Kernel> forward_op =
        functions::primal::unary::pointwise<Kernel>; ///< Function pointer to
                                                     ///< the value operation.

    functions::adjoint::unary::pointwise_fn<Kernel> backward_op =
        functions::adjoint::unary::pointwise<
            Kernel>; ///< Function pointer to the gradient operation.

    INode *input = nullptr; ///< Pointer to the input Node.

    const Scalar *end =
        nullptr; ///< Pointer to the end of longest buffer (used for iteration,
                 ///< buffer may differ depending on operation).

  public:
    /**
     * @brief Constructs a unary node with the given operation and gradient.
     * @ingroup nodes
     * @param operation  Function pointer to the value operation.
     * @param derivative Function pointer to the gradient operation.
     * @param input_ptr    Pointer to the input node.
     * @param value_shape Shape of the value and gradient tensors.
     */
    Node_unary(functions::primal::unary::pointwise_fn<Kernel> operation,
               functions::adjoint::unary::pointwise_fn<Kernel> derivative,
               INode *input_ptr, std::span<const int> value_shape)
        : INode(value_shape, false), forward_op(operation),
          backward_op(derivative), input(input_ptr) {
        this->end =
            this->value().data() +
            this->value()
                .size(); // Points to end of value if not overriden in operator.
    }

    /**
     * @brief Returns the type of the node as a string.
     * @ingroup nodes
     */
    const char *node_type() const noexcept override { return "Node_unary"; }

    /**
     * @brief Evaluates the unary operation by applying forward_op, if not
     * @ingroup nodes
     * already evaluated.
     */
    inline void eval() override {
        if (!this->evaluated()) {
            this->input->eval();

            forward_op(this->input->value().data(), this->value().data(), end);
            this->evaluated_ = true;
        }
    }

    /**
     * @brief Propagates gradients back through the unary operation, by applying
     * @ingroup nodes
     * backward_op.
     */
    inline void getGrad() override {
        backward_op(this->input->value().data(), this->input->gradient().data(),
                    this->value().data(), this->gradient().data(), end);

        if (!this->input->isInput()) {
            this->input->getGrad();
        }
    }

    friend Node sum(Computation_graph &rec, Node A);
};

} // namespace kaad
