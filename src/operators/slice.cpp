#include <kaad/operators/operators.hpp> // for slice

#include "../exceptions.hpp"          // for ArgumentError, make_graph_e...
#include "../graph/nodes/slice.hpp"   // for NodeSlice
#include <algorithm>                  // for copy, __copy_fn, fill
#include <cstddef>                    // for size_t
#include <initializer_list>           // for initializer_list
#include <kaad/graph/graph.hpp>       // for Graph, slice
#include <kaad/graph/node_handle.hpp> // for Node
#include <kaad/graph/nodes/inode.hpp> // for INode
#include <kaad/tensor/tensor.hpp>     // for Tensor
#include <memory>                     // for unique_ptr, make_unique
#include <span>                       // for span
#include <string>                     // for basic_string, string, to_string
#include <utility>                    // for pair
#include <vector>                     // for vector

namespace kaad {

Node slice(Graph &rec, Node input, std::initializer_list<int> size,
           std::initializer_list<int> offset) {
    std::size_t rec_len = rec.nodes.size();

    INode *input_ptr = rec.get_node(input);
    Tensor &input_val = input_ptr->value();

    if (size.size() > input_val.rank()) {
        std::span<const int> size_span(size.begin(), size.size());
        throw ArgumentError(make_graph_errmsg(
            "argument error", rec_len, "slice",
            "length of size is bigger than A.rank()",
            {{"size", size_span}, {"A.shape", input_val.shape()}}));
    }
    if (offset.size() > input_val.rank()) {
        std::span<const int> offset_span(offset.begin(), offset.size());
        throw ArgumentError(make_graph_errmsg(
            "argument error", rec_len, "slice",
            "length of offset is bigger than A.rank()",
            {{"offset", offset_span}, {"A.shape", input_val.shape()}}));
    }

    int size_diff = static_cast<int>(input_val.rank() - size.size());
    std::vector<int> size_owned(input_val.rank());

    // if length of size is smaller than rank of A, the left out dimensions stay
    // the same.
    std::copy(input_val.shape().begin(), input_val.shape().begin() + size_diff,
              size_owned.begin());
    std::ranges::copy(size, size_owned.begin() + size_diff);

    // if length of offset is smaller than rank of A, the left out offsets are
    // set to 0.
    int offset_diff = static_cast<int>(input_val.rank() - offset.size());
    std::vector<int> offset_owned(input_val.rank());
    std::fill(offset_owned.begin(), offset_owned.begin() + offset_diff, 0);
    std::ranges::copy(offset, offset_owned.begin() + offset_diff);

    for (std::size_t i = 0; i < input_val.rank(); i++) {
        if (offset_owned[i] + size_owned[i] > input_val.shape()[i]) {
            std::span<const int> size_span(size.begin(), size.size());
            std::span<const int> offset_span(offset.begin(), offset.size());
            std::string idx_str = std::to_string(i);
            std::string msg = "offset[";
            msg += idx_str;
            msg += "] + length[";
            msg += idx_str;
            msg += "] > A.shape[";
            msg += idx_str;
            msg += "]";
            throw ArgumentError(make_graph_errmsg(
                "argument error", rec_len, "slice", msg.c_str(),
                {{"size", size_span},
                 {"offset", offset_span},
                 {"A.shape", input_val.shape()}}));
        }
    }

    std::size_t new_len = input_val.rank();
    std::vector<int> new_shape(new_len);
    std::ranges::copy(size_owned, new_shape.begin());

    rec.nodes.push_back(
        std::make_unique<NodeSlice>(input_ptr, offset_owned.data(), new_shape));
    return rec.back_handle();
}

} // namespace kaad
