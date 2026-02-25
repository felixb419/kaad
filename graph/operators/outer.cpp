#include "../nodes/outer.hpp"          // for Node_outer
#include "../../graph/nodes/inode.hpp" // for INode
#include "../../tensor/tensor.hpp"     // for Tensor
#include "../computation_graph.hpp"    // for Computation_graph
#include "../node_handle.hpp"          // for Node_handle
#include "operators.hpp"               // for outer
#include <algorithm>                   // for copy
#include <cstddef>                     // for size_t
#include <memory>                      // for unique_ptr, __u...
#include <utility>                     // for move
#include <vector>                      // for vector

namespace kaad {

Node_handle outer(Computation_graph &rec, Node_handle A, Node_handle B) {
    int recLen = rec.nodes.size();

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
