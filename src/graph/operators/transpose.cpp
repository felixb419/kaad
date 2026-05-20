#include "../../graph/operator_node.hpp"
#include "../../operations/transpose.hpp"
#include "kaad/graph/internal/inode.hpp"

#include <array>
#include <cstddef>
#include <kaad/graph/graph.hpp>
#include <kaad/graph/node_handle.hpp>
#include <kaad/static_vector.hpp>
#include <vector>

namespace kaad {

Node Graph::transpose(Node input, StaticVector<std::size_t> perm) {

    INode *inp_ptr = this->get_node(input);

    if (perm.empty()) {

        perm.resize(inp_ptr->rank());

        std::size_t val = inp_ptr->rank();
        for (std::size_t &elem : perm) {
            elem = --val;
        }
    }

    this->nodes.push_back(new OperatorNode<operations::Transpose>(
        std::array{this->get_node(input)},
        operations::Transpose::Metadata(perm)));

    return Node(this->nodes.size() - 1, this);
}

} // namespace kaad
