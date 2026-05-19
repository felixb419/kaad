#include "../operations/slice.hpp"

#include "../graph/operator_node.hpp"
#include "kaad/tensor/internal/tensor_types.hpp"

#include <array>
#include <cstddef>
#include <kaad/graph/graph.hpp>
#include <kaad/graph/node_handle.hpp>
#include <kaad/operators/operators.hpp>
#include <kaad/static_vector.hpp>
#include <memory>
#include <vector>

namespace kaad {

Node slice(Graph &rec, Node input, Shape shape,
           StaticVector<std::size_t> start) {

    rec.nodes.push_back(std::make_unique<OperatorNode<operations::Slice>>(
        std::array{rec.get_node(input)},
        operations::Slice::Metadata{shape, start}));

    return rec.back_handle();
}

} // namespace kaad
