#pragma once

#include "../../tensor/tensor.hpp"           // for Tensor
#include "../../tensorfuncs/adjoint_ops.hpp" // for tensorfuncs::adjoint
#include "../../tensorfuncs/kernels.hpp"     // for Kernels
#include "../../tensorfuncs/primal_ops.hpp"  // for tensorfuncs::primal
#include <algorithm>                         // for std::copy, std::fill
#include <memory>                            // for std::make_unique

namespace kaad {

template <typename T> class Computation_graph;
template <typename T> class INode;
template <typename T> class Node_handle;
template <typename T, class Kernel> class Node_unary;

/**
 * @brief Contains a collection of unary functions for pointwise version of the
 * operation and gradient of a given unary Kernel.
 *
 * @tparam T Datatype the operations are performed on (e.g. float, double, ...).
 * @tparam Kernel Kernel the functions should be using.
 */
template <typename T, class Kernel> struct UnaryKernels {
    tensorfuncs::primal::unary::pointwise_fn<T, Kernel> op =
        tensorfuncs::primal::unary::pointwise<T, Kernel>;
    tensorfuncs::adjoint::unary::pointwise_fn<T, Kernel> grad =
        tensorfuncs::adjoint::unary::pointwise<T, Kernel>;
};

/**
 * @internal
 * @brief Internal helper function not intended for direct user calls.
 *
 * Adds a generalized unary operation node to the computation graph `rec`.
 * Applies the unary operation specified by `kernels` to the input tensor node
 * `A`.
 *
 * @tparam T The data type of tensor elements.
 * @tparam Kernel The kernel providing forward operation and gradient.
 *
 * @param rec Reference to the computation graph.
 * @param A Handle of the input node.
 * @param kernels Unary operation and gradient kernels.
 * @return Handle of the newly created unary operation node.
 */
template <typename T, class Kernel>
Node_handle<T> unOperator(Computation_graph<T> &rec, Node_handle<T> A,
                          const UnaryKernels<T, Kernel> kernels) {
    INode<T> *A_ptr = rec.get_node(A);
    Tensor &A_val = A_ptr->value;

    rec.nodes.push_back(std::move(std::make_unique<Node_unary<T, Kernel>>(
        kernels.op, kernels.grad, A_ptr, A_val.shape())));

    return rec.back_handle();
}

/**
 * @brief Adds a unary negation node (-A) to the computation graph.
 *
 * @tparam T The data type of the tensor values.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A Handle of the input tensor node A.
 * @return A handle of the new node representing the negated tensor,
 *         with the same shape as A.
 */
template <typename T>
Node_handle<T> negative(Computation_graph<T> &rec, Node_handle<T> A) {
    static const UnaryKernels<T, class Kernels::Neg<T>> negK;
    return unOperator(rec, A, negK);
}

/**
 * @brief Adds a unary square node (A²) to the computation graph.
 *
 * @tparam T The data type of the tensor values.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A Handle of the input tensor node A.
 * @return A handle of the new node representing the element-wise square of A,
 *         with the same shape as the input tensor.
 */
template <typename T>
Node_handle<T> square(Computation_graph<T> &rec, Node_handle<T> A) {
    static const UnaryKernels<T, class Kernels::Square<T>> squareK;
    return unOperator(rec, A, squareK);
}

/**
 * @brief Adds a unary square root node (√A) to the computation graph.
 *
 * @tparam T The data type of the tensor values.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A Handle of the input tensor node A.
 * @return A handle of the new node representing the element-wise square root
 * of A, with the same shape as the input tensor.
 */
template <typename T>
Node_handle<T> sqrt(Computation_graph<T> &rec, Node_handle<T> A) {
    static const UnaryKernels<T, class Kernels::Sqrt<T>> sqrtK;
    return unOperator(rec, A, sqrtK);
}

/**
 * @brief Adds a unary logarithm node (log(A)) to the computation graph.
 *
 * @tparam T The data type of the tensor values.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A Handle of the input tensor node A.
 * @return A handle of the new node representing the element-wise logarithm
 * of A, with the same shape as the input tensor.
 */
template <typename T>
Node_handle<T> log(Computation_graph<T> &rec, Node_handle<T> A) {
    static const UnaryKernels<T, class Kernels::Log<T>> logK;
    return unOperator(rec, A, logK);
}

/**
 * @brief Adds a unary exponent node (e^A) to the computation graph.
 *
 * @tparam T The data type of the tensor values.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A Handle of the input tensor node A.
 * @return A handle of the new node representing the element-wise exponent
 * of A, with the same shape as the input tensor.
 */
template <typename T>
Node_handle<T> exp(Computation_graph<T> &rec, Node_handle<T> A) {
    static const UnaryKernels<T, class Kernels::Exp<T>> expK;
    return unOperator(rec, A, expK);
}

/**
 * @brief Adds a unary absolute value node (|A|) to the computation graph.
 *
 * @tparam T The data type of the tensor values.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A Handle of the input tensor node A.
 * @return A handle of the new node representing the element-wise absolute
 * value of A, with the same shape as the input tensor.
 */
template <typename T>
Node_handle<T> abs(Computation_graph<T> &rec, Node_handle<T> A) {
    static const UnaryKernels<T, class Kernels::Abs<T>> absK;
    return unOperator(rec, A, absK);
}

} // namespace kaad
