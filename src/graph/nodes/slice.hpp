#pragma once

#include <cstddef>                      // for size_t
#include <kaad/functions/adjoint.hpp>   // for slice, slice_fn
#include <kaad/functions/primal.hpp>    // for slice, slice_fn
#include <kaad/graph/nodes/inode.hpp>   // for INode
#include <kaad/scalar.hpp>              // for Scalar
#include <kaad/static_vector.hpp>       // for StaticVector
#include <kaad/tensor/tensor_types.hpp> // for Stride, ShapeView

namespace kaad {

/**
 * @brief A slice operation node for a @ref kaad::Graph
 * @ingroup nodes
 * @internal
 */
class NodeSlice : public INode {
  private:
    INode *input = nullptr; ///< Pointer to the input Node.

    functions::primal::unary::slice_fn<Scalar> forward_op =
        functions::primal::unary::slice;
    functions::adjoint::unary::slice_fn<Scalar> backward_op =
        functions::adjoint::unary::slice;

    Stride input_stride; ///< Stride array for tensor input.
    Stride value_stride; ///< Stride array for tensor value.
    StaticVector<std::size_t>
        start_offset_a; ///< Offset for the start of input.
    StaticVector<std::size_t>
        value_offset; ///< Per-dim offset to the end of value buffer.
    std::size_t value_rank =
        0; ///< Number of the dimensions of the value tensor.

    void metadata(const int *offset_arr);

  public:
    /**
     * @brief Construct slice node.
     * @param input_ptr Pointer to the first input node.
     * @param offset_arr Array with the offset of the start of @c value.
     * @param value_shape Output/gradient shape
     */
    NodeSlice(INode *input_ptr, const int *offset_arr, ShapeView value_shape);

    /// @return Type of the node as a string.
    [[nodiscard]] const char *node_type() const noexcept override;

    /// Compute @c value for this node.
    /// Computes @c value for @c lhs and @c rhs first.
    void eval() override;

    /// Compute @c gradient for this node.
    /// Computes @c gradient for @c lhs and @c rhs after.
    void get_grad() override;
};

} // namespace kaad
