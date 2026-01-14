#pragma once

#include "../../tensor/tensor.hpp" // for Tensor
#include "../../utils.hpp"         // for print_arr
#include "exceptions.hpp"          // for shape_error, argument_error
#include <initializer_list>        // for std::initializer_list
#include <memory>                  // for std::make_unique

namespace kaad {

template <typename T> struct Computation_graph;
template <typename T> struct INode;
template <typename T> struct Node_transp;

/**
 * @brief Adds a unary transpose node to the computation graph.
 *
 * Transposes the input tensor node `A_ptr` according to the permutation `perm`.
 * If `perm` is empty, the tensor is fully transposed by reversing its
 * dimensions.
 *
 * @tparam T The data type of the tensor values.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A_ptr Pointer to the input tensor node A.
 * @param perm An optional initializer list specifying the permutation of axes.
 *             If not provided or empty, a full transpose (reverse of all axes)
 * is performed.
 * @return A pointer to the new node representing the transposed tensor,
 *         with shape adjusted according to `perm` or full transpose.
 */
template <typename T>
INode<T> *transpose(Computation_graph<T> &rec, INode<T> *A_ptr,
                    std::initializer_list<int> perm = {}) {
    int recLen = rec.nodes.size();
    Tensor<T> &A = A_ptr->value;

    if (A.nDims() < 2) {
        throw shape_error(recLen, "transpose", "A.nDims() hast to be > 1",
                          {{"A.shape", A.shape}});
    }

    std::vector<int> shape_T(A.nDims());
    std::vector<int> stride_T(A.nDims());
    if (perm.size() == 0) {
        std::copy(A.shape.begin(), A.shape.end(), shape_T.begin());
        std::copy(A.stride.begin(), A.stride.end(), stride_T.begin());

        transp(A.shape.data(), A.stride.data(), A.nDims(), shape_T.data(),
               stride_T.data());
    } else {
        if (perm.size() != A.nDims()) {
            throw argument_error(recLen, "transpose",
                                 "perm.size() has to be same as A.nDims()",
                                 {{"A.shape", A.shape}});
        }

        int *count = new int[A.nDims()];
        std::fill(count, count + A.nDims(), 0);

        int *sh = shape_T.data();
        int *st = stride_T.data();
        for (int idx : perm) {

            count[idx]++;

            *(sh++) = A.shape[idx];
            *(st++) = A.stride[idx];
        }
        for (int *p = count; p != count + A.nDims(); p++) {
            if (*p != 1) {
                // change to print perm instead of A.shape
                throw argument_error(
                    recLen, "transpose",
                    "perm has to contain index of every dimension "
                    "exactly once",
                    {{"A.shape", A.shape}});
            }
        }
    }

    auto newNode =
        std::make_unique<Node_transp<T>>(A_ptr, shape_T, stride_T, A.nDims());
    auto raw_ptr = newNode.get();
    newNode->A_end = A.data() + A.nElems();
    newNode->C_end = raw_ptr->value.data() + raw_ptr->value.nElems();
    rec.nodes.push_back(std::move(newNode));

    return raw_ptr;
}

} // namespace kaad
