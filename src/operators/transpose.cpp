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
#include <kaad/tensor/tensor_types.hpp> // for Shape, Stride
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

    INode *input_ptr = rec.get_node(input);
    TensorViewConst input_val = input_ptr->value().view();

    if (input_val.rank() < 2) {
        throw ShapeError(make_graph_errmsg(rec_len, "transpose",
                                           "A.rank() hast to be > 1",
                                           {{"A.shape", input_val.shape}}));
    }

    Shape shape_buff;
    Stride stride_buff;
    TensorViewConst value_t;
    if (perm.empty()) {

        value_t = input_val.transpose(shape_buff, stride_buff);

    } else {
        if (perm.size() != input_val.rank()) {
            throw ArgumentError(make_graph_errmsg(
                rec_len, "transpose", "perm.size() has to be same as A.rank()",
                {{"A.shape", input_val.shape}}));
        }

        if (contains_duplicates(perm)) {
            throw ArgumentError(make_graph_errmsg(
                rec_len, "transpose",
                "perm has to contain index of every dimension exactly once",
                {{"perm", perm}}));
        }

        value_t = input_val.transpose(shape_buff, stride_buff, perm);
    }

    rec.nodes.push_back(
        std::make_unique<NodeTransp>(input_ptr, value_t.shape, value_t.stride));

    return rec.back_handle();
}

} // namespace kaad
