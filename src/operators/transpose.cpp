#include "../operations/transpose.hpp"

#include "../graph/operator_node.hpp"
#include "kaad/graph/internal/inode.hpp"

#include <array>
#include <cstddef>
#include <kaad/graph/graph.hpp>
#include <kaad/graph/node_handle.hpp>
#include <kaad/operators/operators.hpp>
#include <kaad/static_vector.hpp>
#include <memory>
#include <vector>

namespace kaad {

Node transpose(Graph &graph, Node input, StaticVector<std::size_t> perm) {

    INode *inp_ptr = graph.get_node(input);

    if (perm.empty()) {

        perm.resize(inp_ptr->rank());

        std::size_t val = inp_ptr->rank();
        for (std::size_t &elem : perm) {
            elem = --val;
        }
    }

    graph.nodes.push_back(std::make_unique<OperatorNode<operations::Transpose>>(
        std::array{graph.get_node(input)},
        operations::Transpose::Metadata(perm)));

    return graph.back_handle();
}

} // namespace kaad
