#pragma once

#include <cstdint>
#include <kaad/functions/adjoint.hpp> // for pointwise_fn, pointwise
#include <kaad/functions/pointwise.hpp>
#include <kaad/functions/primal.hpp>  // for pointwise_fn, pointwise
#include <kaad/graph/nodes/inode.hpp> // for INode
#include <kaad/scalar.hpp>            // for Scalar
#include <span>                       // for span

namespace kaad {

// forward declarations for friend declaration
class Graph;
class Node;

/**
 * @brief A unary operation node for a @ref kaad::Graph
 * @ingroup nodes
 * @internal
 * @tparam Kernel A kernel struct providing `Op` and `Grad` types for the
 * operation.
 */
template <class Kernel> class NodeUnary : public INode {
  private:
    functions::Pointwise::Unary::primal_fn<Kernel> forward_op =
        functions::Pointwise::Unary::primal<Kernel>;

    functions::Pointwise::Unary::adjoint_fn<Kernel> backward_op =
        functions::Pointwise::Unary::adjoint<Kernel>;

    INode *input = nullptr; ///< Pointer to the input Node.

    const Scalar *end =
        nullptr; ///< Pointer to the end of longest buffer (used for iteration,
                 ///< buffer may differ depending on operation).

  public:
    /**
     * @brief Construct Unary node.
     * @param operation  Function pointer to the value operation.
     * @param derivative Function pointer to the gradient operation.
     * @param lhs_ptr Pointer to the first input node.
     * @param rhs_ptr Pointer to the second input node.
     * @param value_shape Output/gradient shape
     */
    NodeUnary(INode *input_ptr, ShapeView value_shape)
        : INode(value_shape, false), input(input_ptr) {
        this->end =
            this->value().data() +
            this->value()
                .size(); // Points to end of value if not overriden in operator.

        if (std::ranges::equal(value_shape, SCALAR_SHAPE)) {
            this->forward_op =
                functions::Pointwise::Unary::primal_scalar_res<Kernel>;
            this->backward_op =
                functions::Pointwise::Unary::adjoint_scalar_res<Kernel>;
        }
    }

    /// @return Type of the node as a string.
    [[nodiscard]] const char *node_type() const noexcept override {
        return "NodeUnary";
    }

    /// Compute @c value for this node.
    /// Computes @c value for @c lhs and @c rhs first.
    void eval() override {
        if (!this->evaluated()) {
            this->input->eval();

            forward_op(this->input->value().data(), this->value().data(), end);
            this->evaluated_ = true;
        }
    }

    /// Compute @c gradient for this node.
    /// Computes @c gradient for @c lhs and @c rhs after.
    void get_grad() override {
        backward_op(this->input->value().data(), this->input->gradient().data(),
                    this->value().data(), this->gradient().data(), end);

        if (!this->input->is_input()) {
            this->input->get_grad();
        }
    }

    friend Node sum(Graph &rec, Node input);
};

} // namespace kaad
