#include <kaad/operators/operators.hpp> // for transpose

#include "../graph/nodes/transp.hpp"    // for NodeTransp
#include <algorithm>                    // for __sort_fn, adjacent_find, sort
#include <cstddef>                      // for size_t
#include <kaad/exceptions.hpp>          // for ArgumentError, ShapeError
#include <kaad/graph/graph.hpp>         // for Graph, transpose
#include <kaad/graph/node_handle.hpp>   // for Node
#include <kaad/graph/nodes/inode.hpp>   // for INode
#include <kaad/static_vector.hpp>       // for StaticVector
#include <kaad/tensor/tensor.hpp>       // for Tensor
#include <kaad/tensor/tensor_types.hpp> // for Shape, Strides
#include <kaad/tensor/tensor_view.hpp>  // for TensorViewConst, TensorView
#include <memory>                       // for unique_ptr, make_unique
#include <ranges>                       // for __adjacent_find_fn
#include <span>                         // for span
#include <string>                       // for basic_string
#include <utility>                      // for pair
#include <vector>                       // for vector

namespace kaad {

bool contains_duplicates(StaticVector<int> vals) {
    std::ranges::sort(vals);
    return std::ranges::adjacent_find(vals) != vals.end();
}

Node transpose(Graph &rec, Node input, StaticVector<int> perm) {
    std::size_t rec_len = rec.nodes.size();

    TensorViewConst input_view = input.value().view();
    std::size_t input_rank = input.shape().size();
    ShapeView input_shape = input.shape();

    if (input_rank < 2) {
        throw ShapeError(make_graph_errmsg(rec_len, "transpose",
                                           "input.rank() hast to be > 1",
                                           {{"input.shape()", input_shape}}));
    }

    Shape shape_buff;
    Strides strides_buff;
    TensorViewConst value_t;
    if (perm.empty()) {

        value_t = input_view.transpose(shape_buff, strides_buff);

    } else {
        if (perm.size() != input_rank) {
            throw ArgumentError(
                make_graph_errmsg(rec_len, "transpose",
                                  "perm.size() has to be same as input.rank()",
                                  {{"input.shape()", input_shape}}));
        }

        if (contains_duplicates(perm)) {
            throw ArgumentError(make_graph_errmsg(
                rec_len, "transpose",
                "perm has to contain index of every dimension exactly once",
                {{"perm", perm}}));
        }

        value_t = input_view.transpose(shape_buff, strides_buff, perm);
    }

    rec.nodes.push_back(std::make_unique<NodeTransp>(
        rec.get_node(input), value_t.shape, value_t.strides));

    return rec.back_handle();
}

} // namespace kaad
