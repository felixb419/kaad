#pragma once

#include "../../tensor/tensor.hpp"           // for Tensor
#include "../../tensorfuncs/adjoint_ops.hpp" // for tensorfuncs::adjoint
#include "../../tensorfuncs/kernels.hpp"     // for Kernels::Sum
#include "../../tensorfuncs/strides.hpp"     // for sum_dim
#include "dispatchers.hpp"                   // for get_sumDim
#include "exceptions.hpp"                    // for argument_error
#include <memory>                            // for std::make_unique
#include <sstream>                           // for std::ostringstream

namespace kaad {

template <typename T> struct Computation_graph;
template <typename T> struct INode;
template <typename T, class Kernel> struct Node_unary;

/**
 * @brief Adds a unary sum node to the computation graph.
 *
 * Computes the sum of all elements in the input tensor node `A_ptr`,
 * producing a scalar tensor node containing the total sum.
 *
 * @tparam T The data type of the tensor values.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A_ptr Pointer to the input tensor node A.
 * @return A pointer to the new node representing the scalar sum of all elements
 * of A.
 */
template <typename T>
INode<T> *sum(Computation_graph<T> &rec, INode<T> *A_ptr) {
    int recLen = rec.nodes.size();
    Tensor<T> &A = A_ptr->value;

    using Kernel = class Kernels::Sum<T>;
    using Op = typename Kernel::Op;
    using Grad = typename Kernel::Grad;
    tensorfuncs::primal::unary::pointwise_fn<T, Op> op =
        tensorfuncs::primal::unary::scalarOut<T, Op>;
    tensorfuncs::adjoint::unary::pointwise_fn<T, Grad> grad =
        tensorfuncs::adjoint::unary::scalarOut<T, Grad>;
    auto newNode =
        std::make_unique<Node_unary<T, Kernel>>(op, grad, A_ptr, (T)0);
    newNode->end = A.data() + A.nElems();
    rec.nodes.push_back(std::move(newNode));
    return rec.nodes.back().get();
}

/**
 * @brief Adds a sum node to the computation graph that sums elements along a
 * specified dimension.
 *
 * Computes the sum of elements in the input tensor node `A_ptr` along the given
 * dimension `dim`. The resulting tensor shape depends on the `keepNDims` flag:
 * - If `keepNDims` is false (default), the dimension `dim` is removed from the
 * output shape.
 * - If `keepNDims` is true, the dimension `dim` is retained with size 1.
 *
 * @tparam T The data type of the tensor values.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A_ptr Pointer to the input tensor node A.
 * @param dim The dimension along which to sum.
 * @param keepNDims If true, retains the summed dimension with size 1; if false,
 * removes it.
 * @return A pointer to the new node representing the tensor after summation
 * along the specified dimension.
 */
template <typename T>
INode<T> *sum(Computation_graph<T> &rec, INode<T> *A_ptr, int dim,
              bool keepNDims = false) {
    int recLen = rec.nodes.size();
    Tensor<T> &A = A_ptr->value;

    if (dim < 0 || dim >= A.nDims()) {
        throw argument_error(recLen, "sum",
                             "dim has to be a valid index of A.shape",
                             {{"A.shape", A.shape}}, {{"dim", dim}});
    }

    if (A.nDims() == 1) {
        return sum(rec, A_ptr);
    }

    size_t newLen = A.nDims();
    std::vector<int> newShape(newLen);
    if (keepNDims) {
        std::copy(A.shape.begin(), A.shape.end(), newShape.begin());
        newShape[dim] = 1;

    } else {
        newLen--;
        std::copy(A.shape.begin(), A.shape.begin() + dim, newShape.begin());
        std::copy(A.shape.begin() + dim + 1, A.shape.end(),
                  newShape.begin() + dim);
    }

    auto newNode = std::make_unique<Node_sum_dim<T>>(A_ptr, newShape, newLen);
    auto raw_ptr = newNode.get();
    if (A.nDims() <= KAAD_MAX_NDIMS) {
        raw_ptr->val_func = detail::Dispatchers::get_sumDim<T>()[A.nDims()];
        raw_ptr->grad_func =
            detail::Dispatchers::get_sumDim_grad<T>()[A.nDims()];
    }

    Strides::sum_dim<T>(*raw_ptr, dim);
    rec.nodes.push_back(std::move(newNode));
    return rec.nodes.back().get();
}

} // namespace kaad
