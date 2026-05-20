#include "../operations/slice.hpp"

#include "../graph/operator_node.hpp"
#include "kaad/tensor/internal/tensor_types.hpp"

#include <array>
#include <cstddef>
#include <kaad/graph/graph.hpp>
#include <kaad/graph/node_handle.hpp>
#include <kaad/operators/operators.hpp>
#include <kaad/static_vector.hpp>
#include <vector>

namespace kaad {

Node slice(Graph &graph, Node input, Shape shape,
           StaticVector<std::size_t> start) {

    graph.nodes.push_back(new OperatorNode<operations::Slice>(
        std::array{graph.get_node(input)},
        operations::Slice::Metadata{shape, start}));

    return graph.back_handle();
}

} // namespace kaad
