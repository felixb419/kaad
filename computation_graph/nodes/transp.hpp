#pragma once

#include "../../tensorfuncs/adjoint_ops.hpp" // for tensorfuncs::adjoint
#include "../../tensorfuncs/kernels.hpp"     // for Null::Op
#include "../../tensorfuncs/primal_ops.hpp"  // for tensorfuncs::primal
#include "inode.hpp"                         // for INode

namespace kaad {

/**
 * @brief A transpose operation node in a computation graph.
 * @see tensorfuncs::primal::unary::noop
 * @see tensorfuncs::adjoint::unary::pointwise
 * @tparam T The scalar type.
 */
template <typename T> struct Node_transp : INode<T> {
    tensorfuncs::primal::unary::pointwise_fn<T, Kernels::Null> forward_op =
        tensorfuncs::primal::unary::noop<
            T, Kernels::Null>; ///< Function pointer to the value operation.

    tensorfuncs::adjoint::unary::pointwise_fn<T, Kernels::Sum<T>> backward_op =
        tensorfuncs::adjoint::unary::pointwise<
            T,
            Kernels::Sum<T>>; ///< Function pointer to the gradient operation.

    const T *A_end = nullptr; ///< Pointer to the end of the A buffer.
    const T *C_end = nullptr; ///< Pointer to the end of the C buffer.

    /**
     * @brief Constructs a transpose node with the given operation and gradient.
     *
     * @param A_ptr Pointer to the input node.
     * @param tensor_args Arguments to construct the output tensor.
     */
    template <typename... TensorArgs>
    Node_transp(INode<T> *A_ptr, TensorArgs &&...tensor_args)
        : INode<T>(A_ptr, tensor_args...) {
        INode<T> *base_ptr = static_cast<INode<T> *>(this);
        this->A_end = base_ptr->A->value.data() + base_ptr->A->value.size();
        this->C_end = base_ptr->value.data() + base_ptr->value.size();
    }

    /**
     * @brief Evaluates the transpose operation by applying forward_op, if not
     * already evaluated.
     */
    inline void eval() override {
        if (!this->evaluated) {
            this->A->eval();

            forward_op(this->A->value.data(), this->value.elements_.data(),
                       A_end);
            this->evaluated = true;
        }
    }

    /**
     * @brief Propagates gradients back through the transpose operation, by
     * applying backward_op.
     */
    inline void getGrad() override {
        backward_op(this->A->value.data(), this->A->gradient.elements_.data(),
                    this->value.data(), this->gradient.data(), C_end);

        if (this->A->hasInputs) {
            this->A->getGrad();
        }
    }
};

} // namespace kaad
