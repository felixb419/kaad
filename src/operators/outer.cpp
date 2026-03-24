#include <kaad/operators/operators.hpp> // for outer

#include <algorithm>                  // for copy
#include <cstddef>                    // for size_t
#include <kaad/graph/graph.hpp>       // for Graph, outer
#include <kaad/graph/node_handle.hpp> // for Node
#include <kaad/graph/nodes/inode.hpp> // for INode
#include <kaad/graph/nodes/outer.hpp> // for NodeOuter
#include <kaad/tensor/tensor.hpp>     // for Tensor
#include <memory>                     // for unique_ptr, make_unique
#include <vector>                     // for vector

namespace kaad {

Node outer(Graph &rec, Node lhs, Node rhs) {

    INode *lhs_ptr = rec.get_node(lhs);
    INode *rhs_ptr = rec.get_node(rhs);
    Tensor &lhs_val = lhs_ptr->value();
    Tensor &rhs_val = rhs_ptr->value();

    std::size_t newLen = lhs_val.rank() + rhs_val.rank();
    std::vector<int> newShape(newLen);
    std::copy(lhs_val.shape().begin(), lhs_val.shape().end(), newShape.begin());
    std::copy(rhs_val.shape().begin(), rhs_val.shape().end(),
              newShape.begin() + static_cast<int>(lhs_val.rank()));

    rec.nodes.push_back(
        std::make_unique<NodeOuter>(lhs_ptr, rhs_ptr, newShape));

    return rec.back_handle();
}

} // namespace kaad
