#pragma once

#include <cstddef>                    // for size_t
#include <kaad/functions/adjoint.hpp> // for flexible, flexible_fn
#include <kaad/functions/primal.hpp>  // for flexible, flexible_fn
#include <kaad/graph/dispatchers.hpp> // for MAX_NDIMS, get_flex...
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
    functions::primal::binary::flexible_fn<Kernel> forward_op =
        functions::primal::binary::flexible<Kernel>; ///< Function pointer to
                                                     ///< the value operation.

    functions::adjoint::binary::flexible_fn<Kernel> backward_op =
        functions::adjoint::binary::flexible<
            Kernel>; ///< Function pointer to the gradient operation.

    INode *lhs = nullptr; ///< Pointer to the first input Node.
    INode *rhs = nullptr; ///< Pointer to the second input Node.

    std::vector<int> lhs_stride;   ///< stride Array for A.
    std::vector<int> rhs_stride;   ///< stride Array for B.
    std::vector<int> value_stride; ///< stride Array for C.
    std::vector<std::size_t>
        C_offset;               ///< Per-dim offset to the end of C buffer.
    std::size_t value_rank = 0; ///< Number of the dimensions of the C tensor.

    void metadata() {
        // compute metadata
        Tensor &lhs = this->lhs->value();
        Tensor &rhs = this->rhs->value();
        Tensor &value = this->value();

        this->value_rank = value.rank();
        this->lhs_stride.resize(this->value_rank);
        this->rhs_stride.resize(this->value_rank);
        this->value_stride.resize(this->value_rank);

        int idx;
        int idxA;
        int idxB;
        int idxC;
        for (std::size_t i = 1; i <= this->value_rank; i++) {
            idx = this->value_rank - i;
            idxA = lhs.rank() - i;
            this->lhs_stride[idx] = idxA >= 0 ? lhs.stride()[idxA] : 0;
            idxB = rhs.rank() - i;
            this->rhs_stride[idx] = idxB >= 0 ? rhs.stride()[idxB] : 0;
            idxC = value.rank() - i;
            this->value_stride[idx] = idxC >= 0 ? value.stride()[idxC] : 0;
            // make sure value_stride[idx] is 1 instead of 0 if value.shape[idx]
            // is 1 for traversing in flexible function
            if (this->value_stride[idx] == 0 && value.shape()[idxC] == 1) {
                this->value_stride[idx] = 1;
            }
        }

        this->C_offset.resize(this->value_rank);
        for (std::size_t i = 0; i < this->value_rank; i++) {
            this->C_offset[i] = static_cast<std::size_t>(value.shape()[i]) *
                                this->value_stride[i];
        }

        // assign compile-time recursive function
        if (this->value_rank <= KAAD_MAX_RANK) {
            this->forward_op =
                Dispatchers::get_flexOp<Kernel>()[this->value_rank];
            this->backward_op =
                Dispatchers::get_flexGrad<Kernel>()[this->value_rank];
        }
    }

  public:
    /**
     * @brief Construct binary node for shape flexible operations.
     * @param lhs_ptr Pointer to the first input node.
     * @param rhs_ptr Pointer to the second input node.
     * @param value_shape Output/gradient shape
     */
    NodeBinaryFlex(INode *lhs_ptr, INode *rhs_ptr,
                   std::span<const int> value_shape)
        : INode(value_shape, false), lhs(lhs_ptr), rhs(rhs_ptr) {

        this->metadata();
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
                       this->value().data(), lhs_stride.data(),
                       rhs_stride.data(), value_stride.data(), C_offset.data(),
                       value_rank);
            this->evaluated_ = true;
        }
    }

    /// Compute @c gradient for this node.
    /// Computes @c gradient for @c lhs and @c rhs after.
    void get_grad() override {
        backward_op(this->lhs->value().data(), this->lhs->gradient().data(),
                    this->rhs->value().data(), this->rhs->gradient().data(),
                    this->value().data(), this->gradient().data(),
                    lhs_stride.data(), rhs_stride.data(), value_stride.data(),
                    C_offset.data(), value_rank);

        if (!this->lhs->is_input()) {
            this->lhs->get_grad();
        }
        if (!this->rhs->is_input()) {
            this->rhs->get_grad();
        }
    }
};

} // namespace kaad
