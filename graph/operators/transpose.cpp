#include "operators.hpp"

#include "../../exceptions.hpp"     // for shape_error, argument_error
#include "../../tensor/tensor.hpp"  // for Tensor
#include "../computation_graph.hpp" // for Computation_graph
#include "../node_handle.hpp"       // for Node_handle
#include "../nodes/inode.hpp"       // for INode
#include "../nodes/transp.hpp"      // for Node_transp
#include <algorithm>                // for std::reverse_copy
#include <initializer_list>         // for std::initializer_list
#include <memory>                   // for std::make_unique

namespace kaad {

Node_handle transpose(Computation_graph &rec, Node_handle A,
                      std::initializer_list<int> perm) {
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

        int *count = new int[A_val.rank()];
        std::fill(count, count + A_val.rank(), 0);

        int *sh = shape_T.data();
        int *st = stride_T.data();
        for (int idx : perm) {

            count[idx]++;

            *(sh++) = A_val.shape()[idx];
            *(st++) = A_val.stride()[idx];
        }
        for (int *p = count; p != count + A_val.rank(); p++) {
            if (*p != 1) {
                throw argument_error(make_graph_errmsg(
                    "argument error", recLen, "transpose",
                    "perm has to contain index of every dimension exactly once",
                    {{"perm", perm}}));
            }
        }
    }

    rec.nodes.push_back(
        std::move(std::make_unique<Node_transp>(A_ptr, shape_T, stride_T)));

    return rec.back_handle();
}

} // namespace kaad
