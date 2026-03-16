#include <kaad/operators/operators.hpp> // for transpose

#include <algorithm>                   // for reverse_copy
#include <cstddef>                     // for size_t
#include <initializer_list>            // for initializer_list
#include <kaad/exceptions.hpp>         // for argument_error, make_graph_e...
#include <kaad/graph/graph.hpp>        // for Graph, transpose
#include <kaad/graph/node_handle.hpp>  // for Node
#include <kaad/graph/nodes/inode.hpp>  // for INode
#include <kaad/graph/nodes/transp.hpp> // for Node_transp
#include <kaad/tensor/tensor.hpp>      // for Tensor
#include <memory>                      // for unique_ptr, make_unique
#include <span>                        // for span
#include <string>                      // for basic_string
#include <utility>                     // for pair
#include <vector>                      // for vector

namespace kaad {

Node transpose(Graph &rec, Node input, std::initializer_list<int> perm) {
    std::size_t recLen = rec.nodes.size();

    INode *input_ptr = rec.get_node(input);
    Tensor &input_val = input_ptr->value();

    if (input_val.rank() < 2) {
        throw shape_error(make_graph_errmsg("shape error", recLen, "transpose",
                                            "A.rank() hast to be > 1",
                                            {{"A.shape", input_val.shape()}}));
    }

    std::vector<int> shape_T(input_val.rank());
    std::vector<int> stride_T(input_val.rank());
    if (perm.size() == 0) {

        std::reverse_copy(input_val.shape().begin(), input_val.shape().end(),
                          shape_T.data());
        std::reverse_copy(input_val.stride().begin(), input_val.stride().end(),
                          stride_T.data());

    } else {
        if (perm.size() != input_val.rank()) {
            throw argument_error(
                make_graph_errmsg("argument erro", recLen, "transpose",
                                  "perm.size() has to be same as A.rank()",
                                  {{"A.shape", input_val.shape()}}));
        }

        std::vector<int> count(input_val.rank());

        auto shape_it = shape_T.begin();
        auto stride_it = stride_T.begin();
        for (int idx : perm) {

            count[idx]++;

            *(shape_it++) = input_val.shape()[idx];
            *(stride_it++) = input_val.stride()[idx];
        }
        for (int count_elem : count) {
            if (count_elem != 1) {
                throw argument_error(make_graph_errmsg(
                    "argument error", recLen, "transpose",
                    "perm has to contain index of every dimension exactly once",
                    {{"perm", perm}}));
            }
        }
    }

    rec.nodes.push_back(
        std::make_unique<Node_transp>(input_ptr, shape_T, stride_T));

    return rec.back_handle();
}

} // namespace kaad
