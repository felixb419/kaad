#pragma once

#include "../../tensorfuncs/adjoint_ops.hpp" // for tensorfuncs::adjoint
#include "../../tensorfuncs/primal_ops.hpp"  // for tensorfuncs::primal
#include "../dispatchers.hpp"                // for get_flexOp, get_flexGrad
#include "inode.hpp"                         // for INode, Node_ptr
#include <vector>                            // for std::vector

namespace kaad {

/**
 * @brief A outer prodcut operation node in a computation graph.
 * @see tensorfuncs::primal::binary::flexible
 * @see tensorfuncs::adjoint::binary::flexible
 * @tparam Kernel A kernel struct providing `Op` and `Grad` types for the
 * operation.
 */
class Node_outer : public INode {
  public:
    const char *node_type() const noexcept override { return "Node_outer"; }

    using Kernel = typename Kernels::Mul<Scalar>;

    INode *B = nullptr; ///< Pointer to the second input Node.

    tensorfuncs::primal::binary::flexible_fn<Scalar, Kernel> forward_op =
        tensorfuncs::primal::binary::flexible<
            Scalar, Kernel>; ///< Function pointer to the value operation.
    tensorfuncs::adjoint::binary::flexible_fn<Scalar, Kernel> backward_op =
        tensorfuncs::adjoint::binary::flexible<
            Scalar,
            Kernel>; ///< Function pointer to the gradient operation.

    std::vector<int> strideA;     ///< stride Array for A.
    std::vector<int> strideB;     ///< stride Array for B.
    std::vector<int> strideC;     ///< stride Array for C.
    std::vector<size_t> C_offset; ///< Per-dim offset to the end of C buffer.
    size_t C_nDims = 0;           ///< Number of the dimensions of the C tensor.

    /**
     * @brief Constructs a outer prodcut operation node with outer prodcut
     * operation and gradient.
     * @param A_ptr Pointer to the first input node.
     * @param B_ptr Pointer to the second input node.
     * @param tensor_args Arguments to construct the output tensor.
     */
    template <typename... TensorArgs>
    Node_outer(INode *A_ptr, INode *B_ptr, TensorArgs &&...tensor_args)
        : B(B_ptr), INode(A_ptr, tensor_args...) {
        // compute metadata
        Tensor_view A = this->A->value.view();
        Tensor_view B = this->B->value.view();
        Tensor_view C = this->value.view();

        this->C_nDims = C.nDims;

        this->strideA.resize(this->C_nDims);
        this->strideB.resize(this->C_nDims);
        this->strideC.resize(this->C_nDims);

        std::copy(C.stride, C.stride + C.nDims, this->strideC.data());
        std::copy(A.stride, A.stride + A.nDims, this->strideA.data());
        std::copy(B.stride, B.stride + B.nDims, this->strideB.data() + A.nDims);

        this->C_offset.resize(this->C_nDims);
        for (int i = 0; i < this->C_nDims; i++) {
            this->C_offset[i] = C.shape[i] * this->strideC[i];
        }

        // assign compile-time recursive function
        if (C_nDims <= Dispatchers::MAX_NDIMS) {
            forward_op = Dispatchers::get_flexOp<Scalar, Kernel>()[C_nDims];
            backward_op = Dispatchers::get_flexGrad<Scalar, Kernel>()[C_nDims];
        }
    }

    /**
     * @brief Evaluates the outer prodcut operation by calling forward_op, if
     * not already evaluated.
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
     * @brief Propagates gradients back through the outer prodcut operation, by
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
