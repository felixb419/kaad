#pragma once

#include "../../tensorfuncs/adjoint_ops.hpp" // for tensorfuncs::adjoint
#include "../../tensorfuncs/primal_ops.hpp"  // for tensorfuncs::primal
#include "inode.hpp"                         // for INode

namespace kaad {

/**
 * @brief A unary operation node in a computation graph.
 * @see tensorfuncs::primal::unary::pointwise
 * @see tensorfuncs::adjoint::unary::pointwise
 * @tparam T The scalar type.
 * @tparam Kernel A kernel struct providing `Op` and `Grad` types for the
 * operation.
 */
template <typename T, class Kernel> class Node_unary : public INode<T> {
  public:
    tensorfuncs::primal::unary::pointwise_fn<T, Kernel> forward_op =
        tensorfuncs::primal::unary::pointwise; ///< Function pointer to the
                                               ///< value operation.

    tensorfuncs::adjoint::unary::pointwise_fn<T, Kernel> backward_op =
        tensorfuncs::adjoint::unary::pointwise; ///< Function pointer to the
                                                ///< gradient operation.

    const T *end =
        nullptr; ///< Pointer to the end of longest buffer (used for iteration,
                 ///< buffer may differ depending on operation).

    /**
     * @brief Constructs a unary node with the given operation and gradient.
     * @param operation  Function pointer to the value operation.
     * @param derivative Function pointer to the gradient operation.
     * @param A_ptr    Pointer to the input node.
     * @param tensor_args       Arguments to construct the output tensor.
     */
    template <typename... TensorArgs>
    Node_unary(tensorfuncs::primal::unary::pointwise_fn<T, Kernel> operation,
               tensorfuncs::adjoint::unary::pointwise_fn<T, Kernel> derivative,
               INode<T> *A_ptr, TensorArgs &&...tensor_args)
        : forward_op(operation), backward_op(derivative),
          INode<T>(A_ptr, tensor_args...) {
        INode<T> *base_ptr = static_cast<INode<T> *>(this);
        this->end = base_ptr->value.data() + base_ptr->value.size();
    }

    /**
     * @brief Evaluates the unary operation by applying forward_op, if not
     * already evaluated.
     */
    inline void eval() override {
        if (!this->evaluated) {
            this->A->eval();

            forward_op(this->A->value.data(), this->value.elements_.data(),
                       end);
            this->evaluated = true;
        }
    }

    /**
     * @brief Propagates gradients back through the unary operation, by applying
     * backward_op.
     */
    inline void getGrad() override {
        backward_op(this->A->value.data(), this->A->gradient.elements_.data(),
                    this->value.data(), this->gradient.data(), end);

        if (this->A->hasInputs) {
            this->A->getGrad();
        }
    }
};

} // namespace kaad
