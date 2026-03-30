#include <kaad/operators/operators.hpp> // for slice

#include "../graph/nodes/slice.hpp"     // for NodeSlice
#include <algorithm>                    // for copy, __copy_fn, fill
#include <cstddef>                      // for size_t
#include <kaad/exceptions.hpp>          // for ArgumentError, make_graph_er...
#include <kaad/graph/graph.hpp>         // for Graph, slice
#include <kaad/graph/node_handle.hpp>   // for Node
#include <kaad/graph/nodes/inode.hpp>   // for INode
#include <kaad/static_vector.hpp>       // for StaticVector
#include <kaad/tensor/tensor.hpp>       // for Tensor
#include <kaad/tensor/tensor_types.hpp> // for Shape
#include <memory>                       // for unique_ptr, make_unique
#include <span>                         // for span
#include <string>                       // for basic_string, string, to_string
#include <utility>                      // for pair
#include <vector>                       // for vector

namespace kaad {

Node slice(Graph &rec, Node input, StaticVector<int> start,
           StaticVector<int> size) {
    std::size_t rec_len = rec.nodes.size();

    INode *input_ptr = rec.get_node(input);
    Tensor &input_val = input_ptr->value();

    if (size.size() > input_val.rank()) {
        throw ArgumentError(make_graph_errmsg(
            rec_len, "slice", "length of size is bigger than A.rank()",
            {{"size", size.view()}, {"A.shape", input_val.shape()}}));
    }
    if (start.size() > input_val.rank()) {
        throw ArgumentError(make_graph_errmsg(
            rec_len, "slice", "length of start is bigger than A.rank()",
            {{"start", start.view()}, {"A.shape", input_val.shape()}}));
    }

    int size_diff = static_cast<int>(input_val.rank() - size.size());
    StaticVector<int> size_owned(input_val.rank());

    // if length of size is smaller than rank of A, the left out dimensions stay
    // the same.
    std::copy(input_val.shape().begin(), input_val.shape().begin() + size_diff,
              size_owned.begin());
    std::ranges::copy(size, size_owned.begin() + size_diff);

    // if length of start is smaller than rank of A, the left out starts are
    // set to 0.
    int start_diff = static_cast<int>(input_val.rank() - start.size());
    StaticVector<int> start_owned(input_val.rank());
    std::fill(start_owned.begin(), start_owned.begin() + start_diff, 0);
    std::ranges::copy(start, start_owned.begin() + start_diff);

    for (std::size_t i = 0; i < input_val.rank(); i++) {
        if (start_owned[i] + size_owned[i] > input_val.shape()[i]) {
            std::span<const int> size_span(size.begin(), size.size());
            std::span<const int> start_span(start.begin(), start.size());
            std::string idx_str = std::to_string(i);
            std::string msg = "offset[";
            msg += idx_str;
            msg += "] + length[";
            msg += idx_str;
            msg += "] > A.shape[";
            msg += idx_str;
            msg += "]";
            throw ArgumentError(
                make_graph_errmsg(rec_len, "slice", msg.c_str(),
                                  {{"size", size_span},
                                   {"offset", start.view()},
                                   {"A.shape", input_val.shape()}}));
        }
    }

    std::size_t new_len = input_val.rank();
    Shape new_shape(new_len);
    std::ranges::copy(size_owned, new_shape.begin());

    rec.nodes.push_back(
        std::make_unique<NodeSlice>(input_ptr, start_owned.data(), new_shape));
    return rec.back_handle();
}

} // namespace kaad
