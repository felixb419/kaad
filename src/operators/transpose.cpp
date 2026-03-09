#include "../../include/kaad/exceptions.hpp" // for make_graph_errmsg, argument_error
#include "../../include/kaad/graph/graph.hpp"         // for Graph, transpose
#include "../../include/kaad/graph/node_handle.hpp"   // for Node
#include "../../include/kaad/graph/nodes/inode.hpp"   // for INode
#include "../../include/kaad/graph/nodes/transp.hpp"  // for Node_transp
#include "../../include/kaad/operators/operators.hpp" // for transpose
#include "../../include/kaad/tensor/tensor.hpp"       // for Tensor
#include <algorithm>                                  // for reverse_copy, fill
#include <initializer_list>                           // for initializer_list
#include <memory>  // for unique_ptr, __unique_ptr_t, make...
#include <utility> // for move
#include <vector>  // for vector

namespace kaad {

Node transpose(Graph &rec, Node A, std::initializer_list<int> perm) {
    int recLen = rec.nodes.size();

    INode *A_ptr = rec.get_node(A);
    Tensor &A_val = A_ptr->value();

    if (A_val.rank() < 2) {
        throw shape_error(make_graph_errmsg("shape error", recLen, "transpose",
                                            "A.rank() hast to be > 1",
                                            {{"A.shape", A_val.shape()}}));
    }

    std::vector<int> shape_T(A_val.rank());
    std::vector<int> stride_T(A_val.rank());
    if (perm.size() == 0) {

        std::reverse_copy(A_val.shape().begin(), A_val.shape().end(),
                          shape_T.data());
        std::reverse_copy(A_val.stride().begin(), A_val.stride().end(),
                          stride_T.data());

    } else {
        if (perm.size() != A_val.rank()) {
            throw argument_error(
                make_graph_errmsg("argument erro", recLen, "transpose",
                                  "perm.size() has to be same as A.rank()",
                                  {{"A.shape", A_val.shape()}}));
        }

        std::vector<int> count(A_val.rank());

        int *sh = shape_T.data();
        int *st = stride_T.data();
        for (int idx : perm) {

            count[idx]++;

            *(sh++) = A_val.shape()[idx];
            *(st++) = A_val.stride()[idx];
        }
        for (int c : count) {
            if (c != 1) {
                throw argument_error(make_graph_errmsg(
                    "argument error", recLen, "transpose",
                    "perm has to contain index of every dimension exactly once",
                    {{"perm", perm}}));
            }
        }
    }

    rec.nodes.push_back(
        std::make_unique<Node_transp>(A_ptr, shape_T, stride_T));

    return rec.back_handle();
}

} // namespace kaad
