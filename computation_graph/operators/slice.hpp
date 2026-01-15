#pragma once

#include "../../tensor/tensor.hpp"       // for Tensor
#include "../../tensorfuncs/strides.hpp" // for slice
#include "dispatchers.hpp"               // for get_slice
#include "exceptions.hpp"                // for argument_error
#include <memory>                        // for std::make_unique
#include <span>                          // for  std::span
#include <string>                        // for  std::string

namespace kaad {

template <typename T> struct Computation_graph;
template <typename T> struct INode;
template <typename T> struct Node_slice;

/**
 * @brief Adds a slice node to the computation graph.
 *
 * Extracts a slice from the input tensor node `A_ptr`, starting at the
 * specified `offset` and extending for the given `size` along each dimension.
 *
 * @tparam T The data type of the tensor values.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A_ptr Pointer to the input tensor node A.
 * @param size An initializer list specifying the size (length) of the slice
 * along each dimension.
 * @param offset An initializer list specifying the starting indices (offset)
 * for the slice along each dimension.
 * @return A pointer to the new node representing the sliced tensor.
 */
template <typename T>
INode<T> *slice(Computation_graph<T> &rec, INode<T> *A_ptr,
                std::initializer_list<int> size,
                std::initializer_list<int> offset) {
    int recLen = rec.nodes.size();
    Tensor<T> &A = A_ptr->value;

    if (size.size() > A.nDims()) {
        std::span<const int> size_span(size.begin(), size.size());
        throw argument_error(recLen, "slice",
                             "length of size is bigger than A.nDims()",
                             {{"size", size_span}, {"A.shape", A.shape()}});
    }
    if (offset.size() > A.nDims()) {
        std::span<const int> offset_span(offset.begin(), offset.size());
        throw argument_error(recLen, "slice",
                             "length of offset is bigger than A.nDims()",
                             {{"offset", offset_span}, {"A.shape", A.shape()}});
    }

    int diff = A.nDims() - offset.size();
    std::vector<int> size_owned(A.nDims());
    std::copy(A.shape_begin(), A.shape_begin() + diff, size_owned.begin());
    std::copy(size.begin(), size.begin() + size.size(),
              size_owned.begin() + diff);

    std::vector<int> offset_owned(A.nDims());
    std::fill(offset_owned.begin(), offset_owned.begin() + diff, 0);
    std::copy(offset.begin(), offset.begin() + offset.size(),
              offset_owned.begin() + diff);

    for (int i = 0; i < A.nDims(); i++) {
        if (offset_owned[i] + size_owned[i] > A.shape()[i]) {
            std::span<const int> size_span(size.begin(), size.size());
            std::span<const int> offset_span(offset.begin(), offset.size());
            std::string idx_str = std::to_string(i);
            std::string msg = "offset[" + idx_str + "] + length[" + idx_str +
                              "] > A.shape[" + idx_str + "]";
            throw argument_error(recLen, "slice", msg.c_str(),
                                 {{"size", size_span},
                                  {"offset", offset_span},
                                  {"A.shape", A.shape()}});
        }
    }

    size_t newLen = A.nDims();
    std::vector<int> newShape(newLen);
    std::copy(size_owned.begin(), size_owned.end(), newShape.begin());

    auto newNode = std::make_unique<Node_slice<T>>(A_ptr, newShape, newLen);
    auto raw_ptr = newNode.get();

    if (A.nDims() < KAAD_MAX_NDIMS) {
        raw_ptr->val_func = detail::Dispatchers::get_slice<T>()[A.nDims()];
        raw_ptr->grad_func =
            detail::Dispatchers::get_slice_grad<T>()[A.nDims()];
    }

    Strides::slice(*raw_ptr, offset_owned.data());

    rec.nodes.push_back(std::move(newNode));
    return rec.nodes.back().get();
}

} // namespace kaad
