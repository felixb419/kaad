#pragma once

#include "../../tensorfuncs/adjoint_ops.hpp" // for tensorfuncs::adjoint
#include "../../tensorfuncs/primal_ops.hpp"  // for tensorfuncs::primal
#include "../dispatchers.hpp"                // for get_flexOp, get_flexGrad
#include "../strides.hpp"                    // for Strides::flexible_binary
#include "inode.hpp"                         // for INode, Node_ptr
#include <vector>                            // for std::vector

namespace kaad {

/**
 * @brief A binary_flex operation node in a computation graph.
 * @see tensorfuncs::primal::binary::flexible
 * @see tensorfuncs::adjoint::binary::flexible
 * @tparam T The scalar type.
 * @tparam Kernel A kernel struct providing `Op` and `Grad` types for the
 * operation.
 */
template <typename T, class Kernel> class Node_binary_flex : public INode<T> {
  public:
    const char *node_type() const noexcept override { return "Node_binary_flex"; }

    INode<T> *B = nullptr; ///< Pointer to the second input Node.

    tensorfuncs::primal::binary::flexible_fn<T, Kernel> forward_op =
        tensorfuncs::primal::binary::flexible<
            T, Kernel>; ///< Function pointer to the value operation.
    tensorfuncs::adjoint::binary::flexible_fn<T, Kernel> backward_op =
        tensorfuncs::adjoint::binary::flexible<
            T, Kernel>; ///< Function pointer to the gradient operation.

    std::vector<int> strideA;     ///< stride Array for A.
    std::vector<int> strideB;     ///< stride Array for B.
    std::vector<int> strideC;     ///< stride Array for C.
    std::vector<size_t> C_offset; ///< Per-dim offset to the end of C buffer.
    size_t C_nDims = 0;           ///< Number of the dimensions of the C tensor.

    /**
     * @brief Constructs a binary_flex operation node with binary_flex operation
     * and gradient.
     * @param A_ptr Pointer to the first input node.
     * @param B_ptr Pointer to the second input node.
     * @param tensor_args Arguments to construct the output tensor.
     */
    template <typename... TensorArgs>
    Node_binary_flex(INode<T> *A_ptr, INode<T> *B_ptr,
                     TensorArgs &&...tensor_args)
        : B(B_ptr), INode<T>(A_ptr, tensor_args...) {
        Strides::flexible_binary(*this);

        if (C_nDims <= Dispatchers::MAX_NDIMS) {
            forward_op = Dispatchers::get_flexOp<T, Kernel>()[C_nDims];
            backward_op = Dispatchers::get_flexGrad<T, Kernel>()[C_nDims];
        }
    }

    /**
     * @brief Evaluates the binary_flex operation by calling forward_op, if not
     * already evaluated.
     */
    inline void eval() override {
        if (!this->evaluated) {
            this->A->eval();
            this->B->eval();

            forward_op(this->A->value.data(), this->B->value.data(),
                       this->value.elements_.data(), strideA.data(),
                       strideB.data(), strideC.data(), C_offset.data(),
                       C_nDims);
            this->evaluated = true;
        }
    }

    /**
     * @brief Propagates gradients back through the binary_flex operation, by
     * calling backward_op.
     */
    inline void getGrad() override {
        backward_op(this->A->value.data(), this->A->gradient.elements_.data(),
                    this->B->value.data(), this->B->gradient.elements_.data(),
                    this->value.data(), this->gradient.data(), strideA.data(),
                    strideB.data(), strideC.data(), C_offset.data(), C_nDims);

        if (this->A->hasInputs) {
            this->A->getGrad();
        }
        if (this->B->hasInputs) {
            this->B->getGrad();
        }
    }
};

} // namespace kaad
