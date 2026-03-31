#include <kaad/operators/operators.hpp> // for slice

#include "../graph/nodes/slice.hpp"     // for NodeSlice
#include <cstddef>                      // for size_t
#include <kaad/exceptions.hpp>          // for ArgumentError, to_string
#include <kaad/graph/graph.hpp>         // for Graph, slice
#include <kaad/graph/node_handle.hpp>   // for Node
#include <kaad/graph/nodes/inode.hpp>   // for INode
#include <kaad/static_vector.hpp>       // for StaticVector
#include <kaad/tensor/tensor.hpp>       // for Tensor
#include <kaad/tensor/tensor_types.hpp> // for Shape, ShapeView
#include <memory>                       // for allocator, unique_ptr, make_...
#include <string>                       // for char_traits, basic_string
#include <vector>                       // for vector

namespace kaad {

Node slice(Graph &rec, Node input, Shape size, StaticVector<int> start) {

    std::size_t rec_len = rec.nodes.size();
    INode *input_ptr = rec.get_node(input);
    std::size_t input_rank = input_ptr->value().rank();
    ShapeView input_shape = input_ptr->value().shape();

    if (size.size() != input_rank) {

        throw ArgumentError(make_graph_errmsg(
            rec_len, "slice",
            "length of size must be equal to input.rank(), size=" +
                to_string(size) + ", input.shape()=" + to_string(input_shape)));
    }

    if (start.size() > input_rank) {

        throw ArgumentError(make_graph_errmsg(
            rec_len, "slice",
            "length of start must not be bigger than input.rank(), start=" +
                to_string(start) + ", input.shape" + to_string(input_shape)));
    }

    // pad start with 0s
    if (start.size() < input_rank) {
        start.resize(input_rank);
    }

    // make sure slice is not too large
    for (std::size_t i = 0; i < input_rank; i++) {

        if (size[i] + start[i] > input_shape[i]) {

            throw ArgumentError(make_graph_errmsg(
                rec_len, "slice",
                "dimension " + std::to_string(i) +
                    " of slice is too large with a size of " +
                    std::to_string(size[i]) + " starting at " +
                    std::to_string(start[i])));
        }
    }

    rec.nodes.push_back(
        std::make_unique<NodeSlice>(input_ptr, start.data(), size));
    return rec.back_handle();
}

} // namespace kaad
