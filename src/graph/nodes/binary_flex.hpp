#pragma once

#include <cstddef> // for size_t
#include <cstdint>
#include <kaad/functions/flexible.hpp>
#include <kaad/graph/nodes/inode.hpp> // for INode
#include <kaad/max_rank.hpp>          // for KAAD_MAX_RANK
#include <kaad/tensor/tensor.hpp>     // for Tensor
#include <span>                       // for span
#include <vector>                     // for vector

namespace kaad {

template <class Kernel> class NodeBinaryFlex;

/**
 * @brief A binary flex operation node for a @ref kaad::Graph
 * @ingroup nodes
 * @internal
 * @tparam Kernel A kernel struct providing `Op` and `Grad` types for the
 * operation.
 */
template <class Kernel> class NodeBinaryFlex : public INode {
  private:
    functions::Flexible::primal_fn<Kernel> forward_op;

    functions::Flexible::adjoint_fn<Kernel> backward_op;

    INode *lhs = nullptr;
    INode *rhs = nullptr;

    functions::Flexible::Metadata mdata;

  public:
    /**
     * @brief Construct binary node for shape flexible operations.
     * @param lhs_ptr Pointer to the first input node.
     * @param rhs_ptr Pointer to the second input node.
     * @param value_shape Output/gradient shape
     */
    NodeBinaryFlex(INode *lhs_ptr, INode *rhs_ptr, ShapeView value_shape)
        : INode(value_shape, false), lhs(lhs_ptr), rhs(rhs_ptr) {

        // compute metadata
        this->mdata = functions::Flexible::Metadata(lhs_ptr->value().view(),
                                                    rhs_ptr->value().view(),
                                                    this->value().view());

        // assign compile-time recursive functions
        auto fn_pair =
            functions::Flexible::dispatch<Kernel>(this->value().rank());
        this->forward_op = fn_pair.primal;
        this->backward_op = fn_pair.adjoint;
    }

    /// @return Type of the node as a string.
    [[nodiscard]] const char *node_type() const noexcept override {
        return "NodeBinaryFlex";
    }

    /// Compute @c value for this node.
    /// Computes @c value for @c lhs and @c rhs first.
    void eval() override {
        if (!this->evaluated()) {
            this->lhs->eval();
            this->rhs->eval();

            forward_op(this->lhs->value().data(), this->rhs->value().data(),
                       this->value().data(), this->mdata);
            this->evaluated_ = true;
        }
    }

    /// Compute @c gradient for this node.
    /// Computes @c gradient for @c lhs and @c rhs after.
    void get_grad() override {
        backward_op(this->lhs->value().data(), this->lhs->gradient().data(),
                    this->rhs->value().data(), this->rhs->gradient().data(),
                    this->value().data(), this->gradient().data(), this->mdata);

        if (!this->lhs->is_input()) {
            this->lhs->get_grad();
        }
        if (!this->rhs->is_input()) {
            this->rhs->get_grad();
        }
    }
};

} // namespace kaad
