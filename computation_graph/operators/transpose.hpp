#pragma once

#include "../../tensor/tensor.hpp" // for Tensor
#include "exceptions.hpp"          // for shape_error, argument_error
#include <algorithm>               // for std::reverse_copy
#include <initializer_list>        // for std::initializer_list
#include <memory>                  // for std::make_unique

namespace kaad {

template <typename T> class Computation_graph;
template <typename T> class INode;
template <typename T> class Node_handle;
template <typename T> class Node_transp;

/**
 * @brief Adds a unary transpose node to the computation graph.
 *
 * Transposes the input tensor node `A` according to the permutation `perm`.
 * If `perm` is empty, the tensor is fully transposed by reversing its
 * dimensions.
 *
 * @tparam T The data type of the tensor values.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A Handle of the input tensor node A.
 * @param perm An optional initializer list specifying the permutation of axes.
 *             If not provided or empty, a full transpose (reverse of all axes)
 * is performed.
 * @return A handle of the new node representing the transposed tensor,
 *         with shape adjusted according to `perm` or full transpose.
 */
template <typename T>
Node_handle<T> transpose(Computation_graph<T> &rec, Node_handle<T> A,
                         std::initializer_list<int> perm = {}) {
    int recLen = rec.nodes.size();

    INode<T> *A_ptr = rec.get_node(A);
    Tensor<T> &A_val = A_ptr->value;

    if (A_val.nDims() < 2) {
        throw shape_error(recLen, "transpose", "A.nDims() hast to be > 1",
                          {{"A.shape", A_val.shape()}});
    }

    std::vector<int> shape_T(A_val.nDims());
    std::vector<int> stride_T(A_val.nDims());
    if (perm.size() == 0) {

        std::reverse_copy(A_val.shape_begin(), A_val.shape_end(),
                          shape_T.data());
        std::reverse_copy(A_val.stride_begin(), A_val.stride_end(),
                          stride_T.data());

    } else {
        if (perm.size() != A_val.nDims()) {
            throw argument_error(recLen, "transpose",
                                 "perm.size() has to be same as A.nDims()",
                                 {{"A.shape", A_val.shape()}});
        }

        int *count = new int[A_val.nDims()];
        std::fill(count, count + A_val.nDims(), 0);

        int *sh = shape_T.data();
        int *st = stride_T.data();
        for (int idx : perm) {

            count[idx]++;

            *(sh++) = A_val.shape()[idx];
            *(st++) = A_val.stride()[idx];
        }
        for (int *p = count; p != count + A_val.nDims(); p++) {
            if (*p != 1) {
                throw argument_error(
                    recLen, "transpose",
                    "perm has to contain index of every dimension "
                    "exactly once",
                    {{"perm", perm}});
            }
        }
    }

    rec.nodes.push_back(std::move(std::make_unique<Node_transp<T>>(
        A_ptr, shape_T, stride_T, A_val.nDims())));

    return rec.back_handle();
}

} // namespace kaad
