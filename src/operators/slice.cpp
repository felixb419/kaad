#include <kaad/operators/operators.hpp> // for slice

#include "../graph/operator_node.hpp"            // for OperatorNode
#include "../operations/slice.hpp"               // for Slice
#include <array>                                 // for array
#include <cstddef>                               // for size_t
#include <kaad/graph/graph.hpp>                  // for Graph, slice
#include <kaad/graph/internal/inode.hpp>         // for INode
#include <kaad/graph/node_handle.hpp>            // for Node
#include <kaad/static_vector.hpp>                // for StaticVector
#include <kaad/tensor/internal/tensor_types.hpp> // for Shape
#include <memory>                                // for unique_ptr, make_unique
#include <vector>                                // for vector

namespace kaad {

Node slice(Graph &rec, Node input, Shape shape,
           StaticVector<std::size_t> start) {

    rec.nodes.push_back(std::make_unique<OperatorNode<operations::Slice>>(
        std::array{rec.get_node(input)}, shape, start));

    return rec.back_handle();
}

} // namespace kaad
