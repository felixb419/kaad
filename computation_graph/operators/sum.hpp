#pragma once

#include "../../tensor/tensor.hpp"           // for Tensor
#include "../../tensorfuncs/adjoint_ops.hpp" // for tensorfuncs::adjoint
#include "../../tensorfuncs/kernels.hpp"     // for Kernels::Sum
#include "exceptions.hpp"                    // for argument_error
#include <memory>                            // for std::make_unique

namespace kaad {

template <typename T> class Computation_graph;
template <typename T> class INode;
template <typename T> class Node_handle;
template <typename T, class Kernel> class Node_unary;
template <typename T> class Node_sum_dim;

/**
 * @brief Adds a unary sum node to the computation graph.
 *
 * Computes the sum of all elements in the input tensor node `A`,
 * producing a scalar tensor node containing the total sum.
 *
 * @tparam T The data type of the tensor values.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A Handle of the input tensor node A.
 * @return A handle of the new node representing the scalar sum of all elements
 * of A.
 */
template <typename T>
Node_handle<T> sum(Computation_graph<T> &rec, Node_handle<T> A) {
    int recLen = rec.nodes.size();

    INode<T> *A_ptr = rec.get_node(A);
    Tensor &A_val = A_ptr->value;

    using Kernel = class Kernels::Sum<T>;
    tensorfuncs::primal::unary::pointwise_fn<T, Kernel> op =
        tensorfuncs::primal::unary::scalarOut<T, Kernel>;
    tensorfuncs::adjoint::unary::pointwise_fn<T, Kernel> grad =
        tensorfuncs::adjoint::unary::scalarOut<T, Kernel>;

    rec.nodes.push_back(std::move(
        std::make_unique<Node_unary<T, Kernel>>(op, grad, A_ptr, (T)0)));
    static_cast<Node_unary<T, Kernel> *>(rec.nodes.back().get())->end =
        A_val.data() + A_val.size(); // override end from constructor

    return rec.back_handle();
}

/**
 * @brief Adds a sum node to the computation graph that sums elements along a
 * specified dimension.
 *
 * Computes the sum of elements in the input tensor node `A` along the given
 * dimension `dim`. The resulting tensor shape depends on the `keepNDims` flag:
 * - If `keepNDims` is false (default), the dimension `dim` is removed from the
 * output shape.
 * - If `keepNDims` is true, the dimension `dim` is retained with size 1.
 *
 * @tparam T The data type of the tensor values.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A Handle of the input tensor node A.
 * @param dim The dimension along which to sum.
 * @param keepNDims If true, retains the summed dimension with size 1; if false,
 * removes it.
 * @return A handle of the new node representing the tensor after summation
 * along the specified dimension.
 */
template <typename T>
Node_handle<T> sum(Computation_graph<T> &rec, Node_handle<T> A, int dim,
                   bool keepNDims = false) {
    int recLen = rec.nodes.size();

    INode<T> *A_ptr = rec.get_node(A);
    Tensor &A_val = A_ptr->value;

    if (dim < 0 || dim >= A_val.nDims()) {
        throw argument_error(recLen, "sum",
                             "dim has to be a valid index of A.shape",
                             {{"A.shape", A_val.shape()}}, {{"dim", dim}});
    }

    if (A_val.nDims() == 1) {
        return sum(rec, A);
    }

    size_t newLen = A_val.nDims();
    std::vector<int> newShape(newLen);
    if (keepNDims) {
        std::copy(A_val.shape_begin(), A_val.shape_end(), newShape.begin());
        newShape[dim] = 1;

    } else {
        newLen--;
        std::copy(A_val.shape_begin(), A_val.shape_begin() + dim,
                  newShape.begin());
        std::copy(A_val.shape_begin() + dim + 1, A_val.shape_end(),
                  newShape.begin() + dim);
    }

    rec.nodes.push_back(std::move(
        std::make_unique<Node_sum_dim<T>>(A_ptr, dim, newShape, newLen)));
    return rec.back_handle();
}

} // namespace kaad
