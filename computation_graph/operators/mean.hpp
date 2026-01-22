#pragma once

#include "../../tensor/tensor.hpp" // for Tensor
#include "exceptions.hpp"          // for param_error
#include <memory>                  // for std::make_unique

namespace kaad {

template <typename T> class Computation_graph;
template <typename T> class INode;
template <typename T> class Node_mean;
template <typename T> class Node_mean_dim;

/**
 * @brief Adds a unary mean node to the computation graph.
 *
 * Computes the mean (average) of all elements in the input tensor node `A_ptr`,
 * producing a scalar tensor node containing the total average.
 *
 * @tparam T The data type of the tensor values.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A_ptr Pointer to the input tensor node A.
 * @return A pointer to the new node representing the scalar mean of all
 * elements of A.
 */
template <typename T>
INode<T> *mean(Computation_graph<T> &rec, INode<T> *A_ptr) {
    int recLen = rec.nodes.size();
    Tensor<T> &A = A_ptr->value;

    rec.nodes.push_back(std::move(std::make_unique<Node_mean<T>>(A_ptr, (T)0)));
    return rec.nodes.back().get();
}

/**
 * @brief Adds a mean node to the computation graph that computes the mean along
 * a specified dimension.
 *
 * Computes the mean of elements in the input tensor node `A_ptr` along the
 * given dimension `dim`. The resulting tensor shape depends on the `keepNDims`
 * flag:
 * - If `keepNDims` is false (default), the dimension `dim` is removed from the
 * output shape.
 * - If `keepNDims` is true, the dimension `dim` is retained with size 1.
 *
 * @tparam T The data type of the tensor values.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A_ptr Pointer to the input tensor node A.
 * @param dim The dimension along which to compute the mean.
 * @param keepNDims If true, retains the mean-reduced dimension with size 1; if
 * false, removes it.
 * @return A pointer to the new node representing the tensor after mean
 * reduction along the specified dimension.
 */
template <typename T>
INode<T> *mean(Computation_graph<T> &rec, INode<T> *A_ptr, int dim,
               bool keepNDims = 0) {
    int recLen = rec.nodes.size();
    Tensor<T> &A = A_ptr->value;

    if (dim < 0 || dim >= A.nDims()) {
        throw argument_error(recLen, "mean",
                             "dim has to be a valid index of A.shape",
                             {{"A.shape", A.shape()}}, {{"dim", dim}});
    }

    if (A.nDims() == 1) {
        return mean(rec, A_ptr);
    }

    size_t newLen = A.nDims();
    std::vector<int> newShape(newLen);
    if (keepNDims) {
        std::copy(A.shape_begin(), A.shape_end(), newShape.begin());
        newShape[dim] = 1;

    } else {
        newLen--;
        std::copy(A.shape_begin(), A.shape_begin() + dim, newShape.begin());
        std::copy(A.shape_begin() + dim + 1, A.shape_end(),
                  newShape.begin() + dim);
    }

    rec.nodes.push_back(std::move(
        std::make_unique<Node_mean_dim<T>>(A_ptr, dim, newShape, newLen)));
    return rec.nodes.back().get();
}

} // namespace kaad
