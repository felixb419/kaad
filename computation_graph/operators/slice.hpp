#pragma once

#include "../../tensor/tensor.hpp"  // for Tensor
#include "../computation_graph.hpp" // for Computation_graph
#include "../node_handle.hpp"       // for Node_handle
#include "../nodes/slice.hpp"       // for Node_slice
#include "exceptions.hpp"           // for argument_error
#include <memory>                   // for std::make_unique
#include <span>                     // for  std::span
#include <string>                   // for  std::string

namespace kaad {

/**
 * @brief Adds a slice node to the computation graph.
 *
 * Extracts a slice from the input tensor node `A`, starting at the
 * specified `offset` and extending for the given `size` along each dimension.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A Handle of the input tensor node A.
 * @param size An initializer list specifying the size (length) of the slice
 * along each dimension.
 * @param offset An initializer list specifying the starting indices (offset)
 * for the slice along each dimension.
 * @return A handle of the new node representing the sliced tensor.
 */
Node_handle slice(Computation_graph &rec, Node_handle A,
                  std::initializer_list<int> size,
                  std::initializer_list<int> offset) {
    int recLen = rec.nodes.size();

    INode *A_ptr = rec.get_node(A);
    Tensor &A_val = A_ptr->value;

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

    rec.nodes.push_back(std::move(
        std::make_unique<Node_slice>(A_ptr, offset_owned.data(), newShape)));
    return rec.back_handle();
}

} // namespace kaad
