#include <kaad/operators/operators.hpp> // for transpose

#include "../exceptions.hpp"            // for ArgumentError, ShapeError
#include "../graph/nodes/transp.hpp"    // for NodeTransp
#include <algorithm>                    // for reverse_copy
#include <cstddef>                      // for size_t
#include <initializer_list>             // for initializer_list
#include <kaad/graph/graph.hpp>         // for Graph, transpose
#include <kaad/graph/node_handle.hpp>   // for Node
#include <kaad/graph/nodes/inode.hpp>   // for INode
#include <kaad/static_vector.hpp>       // for StaticVector
#include <kaad/tensor/tensor.hpp>       // for Tensor
#include <kaad/tensor/tensor_types.hpp> // for Shape, Stride
#include <memory>                       // for unique_ptr, make_unique
#include <span>                         // for span
#include <string>                       // for basic_string
#include <utility>                      // for pair
#include <vector>                       // for vector

namespace kaad {

Node transpose(Graph &rec, Node input, std::initializer_list<int> perm) {
    std::size_t rec_len = rec.nodes.size();

    INode *input_ptr = rec.get_node(input);
    Tensor &input_val = input_ptr->value();

    if (input_val.rank() < 2) {
        throw ShapeError(make_graph_errmsg("shape error", rec_len, "transpose",
                                           "A.rank() hast to be > 1",
                                           {{"A.shape", input_val.shape()}}));
    }

    Shape shape_t(input_val.rank());
    Stride stride_t(input_val.rank());
    if (perm.size() == 0) {

        std::reverse_copy(input_val.shape().begin(), input_val.shape().end(),
                          shape_t.data());
        std::reverse_copy(input_val.stride().begin(), input_val.stride().end(),
                          stride_t.data());

    } else {
        if (perm.size() != input_val.rank()) {
            throw ArgumentError(
                make_graph_errmsg("argument erro", rec_len, "transpose",
                                  "perm.size() has to be same as A.rank()",
                                  {{"A.shape", input_val.shape()}}));
        }

        StaticVector<int> count(input_val.rank());

        auto *shape_it = shape_t.begin();
        auto *stride_it = stride_t.begin();
        for (int idx : perm) {

            count[idx]++;

            *(shape_it++) = input_val.shape()[idx];
            *(stride_it++) = input_val.stride()[idx];
        }
        for (int count_elem : count) {
            if (count_elem != 1) {
                throw ArgumentError(make_graph_errmsg(
                    "argument error", rec_len, "transpose",
                    "perm has to contain index of every dimension exactly once",
                    {{"perm", perm}}));
            }
        }
    }

    rec.nodes.push_back(
        std::make_unique<NodeTransp>(input_ptr, shape_t, stride_t));

    return rec.back_handle();
}

} // namespace kaad
