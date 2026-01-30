#pragma once

#include "../../tensor/tensor.hpp" // for Tensor
#include "exceptions.hpp"          // for param_error
#include <memory>                  // for std::make_unique

namespace kaad {

template <typename T> class Computation_graph;
template <typename T> class Node_handle;

/**
 * @brief Adds a unary mean node to the computation graph.
 *
 * Computes the mean (average) of all elements in the input tensor node `A`,
 * producing a scalar tensor node containing the total average.
 *
 * @tparam T The data type of the tensor values.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A Handle of the input tensor node A.
 * @return A handle of the new node representing the scalar mean of all
 * elements of A.
 */
template <typename T>
Node_handle<T> mean(Computation_graph<T> &rec, Node_handle<T> A) {
    int recLen = rec.nodes.size();

    rec.nodes.push_back(
        std::move(std::make_unique<Node_mean>(rec.get_node(A), (T)0)));
    return rec.back_handle();
}

/**
 * @brief Adds a mean node to the computation graph that computes the mean along
 * a specified dimension.
 *
 * Computes the mean of elements in the input tensor node `A` along the
 * given dimension `dim`. The resulting tensor shape depends on the `keepNDims`
 * flag:
 * - If `keepNDims` is false (default), the dimension `dim` is removed from the
 * output shape.
 * - If `keepNDims` is true, the dimension `dim` is retained with size 1.
 *
 * @tparam T The data type of the tensor values.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A Handle of the input tensor node A.
 * @param dim The dimension along which to compute the mean.
 * @param keepNDims If true, retains the mean-reduced dimension with size 1; if
 * false, removes it.
 * @return A handle of the new node representing the tensor after mean
 * reduction along the specified dimension.
 */
template <typename T>
Node_handle<T> mean(Computation_graph<T> &rec, Node_handle<T> A, int dim,
                    bool keepNDims = 0) {
    int recLen = rec.nodes.size();

    INode *A_ptr = rec.get_node(A);
    Tensor &A_val = A_ptr->value;

    if (dim < 0 || dim >= A_val.nDims()) {
        throw argument_error(recLen, "mean",
                             "dim has to be a valid index of A.shape",
                             {{"A.shape", A_val.shape()}}, {{"dim", dim}});
    }

    if (A_val.nDims() == 1) {
        return mean(rec, A);
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
        std::make_unique<Node_mean_dim>(A_ptr, dim, newShape, newLen)));
    return rec.back_handle();
}

} // namespace kaad
