#pragma once

#include "../../tensor/tensor.hpp" // for Tensor
#include "../../utils.hpp"         // for print_arr
#include <initializer_list>        // for std::initializer_list
#include <memory>                  // for std::make_unique
#include <sstream>                 // for std::ostringstream

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
        std::ostringstream errmsg;
        errmsg << "shape error in node[" << recLen
               << "] (transpose), A.nDims() hast to be > 1 (shape1=";
        print_arr(A.shape.data(), A.shape.data() + A.nDims(), errmsg);
        errmsg << ")";
        throw std::invalid_argument(errmsg.str());
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
            std::ostringstream errmsg;
            errmsg << "argument error in node[" << recLen
                   << "] (transpose), perm.size() has to be same as A.nDims() "
                      "(perm=";
            print_arr(perm.begin(), perm.end(), errmsg);
            errmsg << ", shape1=";
            print_arr(A.shape.data(), A.shape.data() + A.nDims(), errmsg);
            errmsg << ")";
            throw std::invalid_argument(errmsg.str());
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
                std::ostringstream errmsg;
                errmsg
                    << "argument error in node[" << recLen
                    << "] (transpose), invalid permutation, perm has to "
                       "contain index of every dimension exactly once (perm=";
                print_arr(perm.begin(), perm.end(), errmsg);
                errmsg << ")";
                throw std::invalid_argument(errmsg.str());
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
