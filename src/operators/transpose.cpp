#include <kaad/operators/operators.hpp> // for transpose

#include "../graph/operator_node.hpp"    // for OperatorNode
#include "../operations/transpose.hpp"   // for Transpose
#include <array>                         // for array
#include <cstddef>                       // for size_t
#include <kaad/graph/graph.hpp>          // for Graph, transpose
#include <kaad/graph/internal/inode.hpp> // for INode
#include <kaad/graph/node_handle.hpp>    // for Node
#include <kaad/static_vector.hpp>        // for StaticVector
#include <memory>                        // for unique_ptr, make_unique
#include <vector>                        // for vector

namespace kaad {

Node transpose(Graph &rec, Node input, StaticVector<std::size_t> perm) {

    INode *inp_ptr = rec.get_node(input);

    if (perm.empty()) {

        perm.resize(inp_ptr->rank());

        std::size_t val = inp_ptr->rank();
        for (std::size_t &elem : perm) {
            elem = --val;
        }
    }

    rec.nodes.push_back(std::make_unique<OperatorNode<operations::Transpose>>(
        std::array{rec.get_node(input)}, perm));

    return rec.back_handle();
}

} // namespace kaad
