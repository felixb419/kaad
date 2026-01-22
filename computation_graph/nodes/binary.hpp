#pragma once

#include "../../tensorfuncs/adjoint_ops.hpp" // for tensorfuncs::adjoint
#include "../../tensorfuncs/primal_ops.hpp"  // for tensorfuncs::primal
#include "inode.hpp"                         // for INode

namespace kaad {

/**
 * @brief A binary operation node in a computation graph.
 * @see tensorfuncs::primal::binary::pointwise
 * @see tensorfuncs::adjoint::binary::pointwise
 * @tparam T The scalar type.
 * @tparam Kernel A kernel struct providing `Op` and `Grad` types for the
 * operation.
 */
template <typename T, class Kernel> class Node_binary : public INode<T> {
  public:
    INode<T> *B = nullptr; ///< Pointer to the second input Node.

    tensorfuncs::primal::binary::pointwise_fn<T, Kernel> forward_op =
        tensorfuncs::primal::binary::pointwise; ///< Function pointer to the
                                                ///< value operation.

    tensorfuncs::adjoint::binary::pointwise_fn<T, Kernel> backward_op =
        tensorfuncs::primal::binary::pointwise; ///< Function pointer to the
                                                ///< gradient operation.

    const T *end =
        nullptr; ///< Pointer to the end of longest buffer (used for iteration,
                 ///< buffer may differ depending on operation).

    /**
     * @brief Constructs a binary operation node with the given operation and
     * gradient.
     * @param operation Function pointer to the value operation.
     * @param derivative Function pointer to the gradient operation.
     * @param A_ptr Pointer to the first input node.
     * @param B_ptr Pointer to the second input node.
     * @param tensor_args Arguments to construct the output tensor.
     */
    template <typename... TensorArgs>
    Node_binary(
        tensorfuncs::primal::binary::pointwise_fn<T, Kernel> operation,
        tensorfuncs::adjoint::binary::pointwise_fn<T, Kernel> derivative,
        INode<T> *A_ptr, INode<T> *B_ptr, TensorArgs &&...tensor_args)
        : B(B_ptr), forward_op(operation), backward_op(derivative),
          INode<T>(A_ptr, tensor_args...) {
        INode<T> *base_ptr = static_cast<INode<T> *>(this);
        this->end = base_ptr->value.data() + base_ptr->value.size();
    }

    /**
     * @brief Evaluates the binary operation by applying forward_op, if not
     * already evaluated.
     */
    inline void eval() override {
        if (!this->evaluated) {
            this->A->eval();
            this->B->eval();

            forward_op(this->A->value.data(), this->B->value.data(),
                       this->value.elements_.data(), end);
            this->evaluated = true;
        }
    }

    /**
     * @brief Propagates gradients back through the binary operation by applying
     * backward_op.
     */
    inline void getGrad() override {
        backward_op(this->A->value.data(), this->A->gradient.elements_.data(),
                    this->B->value.data(), this->B->gradient.elements_.data(),
                    this->value.data(), this->gradient.data(), end);

        if (this->A->hasInputs) {
            this->A->getGrad();
        }
        if (this->B->hasInputs) {
            this->B->getGrad();
        }
    }
};

} // namespace kaad
