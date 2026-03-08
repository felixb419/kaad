#include "../../include/kaad/graph/nodes/slice.hpp"   // for Node_slice
#include "../../include/kaad/exceptions.hpp"          // for make_graph_errmsg
#include "../../include/kaad/graph/graph.hpp"         // for Graph
#include "../../include/kaad/graph/node_handle.hpp"   // for Node
#include "../../include/kaad/graph/nodes/inode.hpp"   // for INode
#include "../../include/kaad/operators/operators.hpp" // for slice
#include "../../include/kaad/tensor/tensor.hpp"       // for Tensor
#include <algorithm>                                  // for copy, fill
#include <cstddef>                                    // for size_t
#include <initializer_list>                           // for initializer_list
#include <memory>                                     // for allocator, uniq...
#include <span>                                       // for span
#include <string>                                     // for char_traits
#include <utility>                                    // for move
#include <vector>                                     // for vector

namespace kaad {

Node slice(Graph &rec, Node A, std::initializer_list<int> size,
           std::initializer_list<int> offset) {
    int recLen = rec.nodes.size();

    INode *A_ptr = rec.get_node(A);
    Tensor &A_val = A_ptr->value();

    if (size.size() > A_val.rank()) {
        std::span<const int> size_span(size.begin(), size.size());
        throw argument_error(make_graph_errmsg(
            "argument error", recLen, "slice",
            "length of size is bigger than A.rank()",
            {{"size", size_span}, {"A.shape", A_val.shape()}}));
    }
    if (offset.size() > A_val.rank()) {
        std::span<const int> offset_span(offset.begin(), offset.size());
        throw argument_error(make_graph_errmsg(
            "argument error", recLen, "slice",
            "length of offset is bigger than A.rank()",
            {{"offset", offset_span}, {"A.shape", A_val.shape()}}));
    }

    int size_diff = A_val.rank() - size.size();
    std::vector<int> size_owned(A_val.rank());

    // if length of size is smaller than rank of A, the left out dimensions stay
    // the same.
    std::copy(A_val.shape().begin(), A_val.shape().begin() + size_diff,
              size_owned.begin());
    std::copy(size.begin(), size.end(), size_owned.begin() + size_diff);

    // if length of offset is smaller than rank of A, the left out offsets are
    // set to 0.
    int offset_diff = A_val.rank() - offset.size();
    std::vector<int> offset_owned(A_val.rank());
    std::fill(offset_owned.begin(), offset_owned.begin() + offset_diff, 0);
    std::copy(offset.begin(), offset.end(), offset_owned.begin() + offset_diff);

    for (std::size_t i = 0; i < A_val.rank(); i++) {
        if (offset_owned[i] + size_owned[i] > A_val.shape()[i]) {
            std::span<const int> size_span(size.begin(), size.size());
            std::span<const int> offset_span(offset.begin(), offset.size());
            std::string idx_str = std::to_string(i);
            std::string msg = "offset[" + idx_str + "] + length[" + idx_str +
                              "] > A.shape[" + idx_str + "]";
            throw argument_error(make_graph_errmsg(
                "argument error", recLen, "slice", msg.c_str(),
                {{"size", size_span},
                 {"offset", offset_span},
                 {"A.shape", A_val.shape()}}));
        }
    }

    std::size_t newLen = A_val.rank();
    std::vector<int> newShape(newLen);
    std::copy(size_owned.begin(), size_owned.end(), newShape.begin());

    rec.nodes.push_back(std::move(
        std::make_unique<Node_slice>(A_ptr, offset_owned.data(), newShape)));
    return rec.back_handle();
}

} // namespace kaad
