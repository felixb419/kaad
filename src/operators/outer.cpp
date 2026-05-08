#include <kaad/operators/operators.hpp> // for outer

#include "../graph/operator_node.hpp" // for OperatorNode
#include "../operations/outer.hpp"    // for OuterProduct
#include <array>                      // for array
#include <kaad/graph/graph.hpp>       // for Graph, outer
#include <kaad/graph/inode.hpp>       // for INode
#include <kaad/graph/node_handle.hpp> // for Node
#include <memory>                     // for unique_ptr, make_unique
#include <vector>                     // for vector

namespace kaad {

Node outer(Graph &rec, Node lhs, Node rhs) {

    rec.nodes.push_back(
        std::make_unique<OperatorNode<operations::OuterProduct>>(
            std::array{rec.get_node(lhs), rec.get_node(rhs)}));

    return rec.back_handle();
}

} // namespace kaad
