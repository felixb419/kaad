#include <kaad/operators/operators.hpp>

#include "../graph/input_node.hpp"

#include <kaad/graph/graph.hpp>
#include <kaad/graph/node_handle.hpp>
#include <kaad/tensor/internal/tensor_types.hpp>

namespace kaad {

Node input(Graph &graph, ShapeView shape) {

    graph.nodes.push_back(new InputNode(shape));

    return graph.back_handle();
}

} // namespace kaad
