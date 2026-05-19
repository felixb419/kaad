#include <kaad/operators/operators.hpp>

#include "../graph/input_node.hpp"

#include <kaad/graph/graph.hpp>
#include <kaad/graph/node_handle.hpp>
#include <kaad/tensor/internal/tensor_types.hpp>
#include <memory>

namespace kaad {

Node input(Graph &rec, ShapeView shape) {

    rec.nodes.push_back(std::make_unique<InputNode>(shape));

    return rec.back_handle();
}

} // namespace kaad
