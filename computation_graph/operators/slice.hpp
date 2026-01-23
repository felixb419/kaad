#pragma once

#include "../../tensor/tensor.hpp" // for Tensor
#include "exceptions.hpp"          // for argument_error
#include <memory>                  // for std::make_unique
#include <span>                    // for  std::span
#include <string>                  // for  std::string

namespace kaad {

template <typename T> class Computation_graph;
template <typename T> class INode;
template <typename T> class Node_handle;
template <typename T> class Node_slice;

/**
 * @brief Adds a slice node to the computation graph.
 *
 * Extracts a slice from the input tensor node `A`, starting at the
 * specified `offset` and extending for the given `size` along each dimension.
 *
 * @tparam T The data type of the tensor values.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A Handle of the input tensor node A.
 * @param size An initializer list specifying the size (length) of the slice
 * along each dimension.
 * @param offset An initializer list specifying the starting indices (offset)
 * for the slice along each dimension.
 * @return A handle of the new node representing the sliced tensor.
 */
template <typename T>
Node_handle<T> slice(Computation_graph<T> &rec, Node_handle<T> A,
                     std::initializer_list<int> size,
                     std::initializer_list<int> offset) {
    int recLen = rec.nodes.size();

    INode<T> *A_ptr = rec.get_node(A);
    Tensor<T> &A_val = A_ptr->value;

    if (size.size() > A_val.nDims()) {
        std::span<const int> size_span(size.begin(), size.size());
        throw argument_error(recLen, "slice",
                             "length of size is bigger than A.nDims()",
                             {{"size", size_span}, {"A.shape", A_val.shape()}});
    }
    if (offset.size() > A_val.nDims()) {
        std::span<const int> offset_span(offset.begin(), offset.size());
        throw argument_error(
            recLen, "slice", "length of offset is bigger than A.nDims()",
            {{"offset", offset_span}, {"A.shape", A_val.shape()}});
    }

    int diff = A_val.nDims() - offset.size();
    std::vector<int> size_owned(A_val.nDims());
    std::copy(A_val.shape_begin(), A_val.shape_begin() + diff,
              size_owned.begin());
    std::copy(size.begin(), size.begin() + size.size(),
              size_owned.begin() + diff);

    std::vector<int> offset_owned(A_val.nDims());
    std::fill(offset_owned.begin(), offset_owned.begin() + diff, 0);
    std::copy(offset.begin(), offset.begin() + offset.size(),
              offset_owned.begin() + diff);

    for (int i = 0; i < A_val.nDims(); i++) {
        if (offset_owned[i] + size_owned[i] > A_val.shape()[i]) {
            std::span<const int> size_span(size.begin(), size.size());
            std::span<const int> offset_span(offset.begin(), offset.size());
            std::string idx_str = std::to_string(i);
            std::string msg = "offset[" + idx_str + "] + length[" + idx_str +
                              "] > A.shape[" + idx_str + "]";
            throw argument_error(recLen, "slice", msg.c_str(),
                                 {{"size", size_span},
                                  {"offset", offset_span},
                                  {"A.shape", A_val.shape()}});
        }
    }

    size_t newLen = A_val.nDims();
    std::vector<int> newShape(newLen);
    std::copy(size_owned.begin(), size_owned.end(), newShape.begin());

    rec.nodes.push_back(std::move(std::make_unique<Node_slice<T>>(
        A_ptr, offset_owned.data(), newShape, newLen)));
    return rec.back_handle();
}

} // namespace kaad
