#include "../../graph/operator_node.hpp"
#include "../../operations/slice.hpp"
#include "kaad/tensor/internal/tensor_types.hpp"

#include <array>
#include <cstddef>
#include <kaad/graph/graph.hpp>
#include <kaad/graph/node_handle.hpp>
#include <kaad/static_vector.hpp>
#include <vector>

namespace kaad {

Node Graph::slice(Node input, Shape shape, StaticVector<std::size_t> start) {

    this->nodes.push_back(new OperatorNode<operations::Slice>(
        std::array{this->get_node(input)},
        operations::Slice::Metadata{shape, start}));

    return Node(this->nodes.size() - 1, this);
}

} // namespace kaad
