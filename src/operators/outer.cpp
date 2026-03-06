#include "../../include/kaad/graph/nodes/outer.hpp"       // for Node_outer
#include "../../include/kaad/graph/computation_graph.hpp" // for Computation_graph
#include "../../include/kaad/graph/node_handle.hpp"       // for Node
#include "../../include/kaad/graph/nodes/inode.hpp"       // for INode
#include "../../include/kaad/operators/operators.hpp"     // for outer
#include "../../include/kaad/tensor/tensor.hpp"           // for Tensor
#include <algorithm>                                      // for copy
#include <cstddef>                                        // for size_t
#include <memory>  // for unique_ptr, __u...
#include <utility> // for move
#include <vector>  // for vector

namespace kaad {

Node outer(Computation_graph &rec, Node A, Node B) {

    INode *A_ptr = rec.get_node(A);
    INode *B_ptr = rec.get_node(B);
    Tensor &A_val = A_ptr->value();
    Tensor &B_val = B_ptr->value();

    std::size_t newLen = A_val.rank() + B_val.rank();
    std::vector<int> newShape(newLen);
    std::copy(A_val.shape().begin(), A_val.shape().end(), newShape.begin());
    std::copy(B_val.shape().begin(), B_val.shape().end(),
              newShape.begin() + A_val.rank());

    rec.nodes.push_back(
        std::move(std::make_unique<Node_outer>(A_ptr, B_ptr, newShape)));

    return rec.back_handle();
}

} // namespace kaad
