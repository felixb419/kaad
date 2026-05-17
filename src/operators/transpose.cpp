#include <kaad/operators/operators.hpp>

#include "../graph/operator_node.hpp"
#include "../operations/transpose.hpp"
#include <array>
#include <cstddef>
#include <kaad/graph/graph.hpp>
#include <kaad/graph/internal/inode.hpp>
#include <kaad/graph/node_handle.hpp>
#include <kaad/static_vector.hpp>
#include <memory>
#include <vector>

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
        std::array{rec.get_node(input)},
        operations::Transpose::Metadata(perm)));

    return rec.back_handle();
}

} // namespace kaad
