#include <kaad/operators/operators.hpp> // for slice

#include <array>                        // for array
#include <cstddef>                      // for size_t
#include <kaad/graph/graph.hpp>         // for Graph, slice
#include <kaad/graph/inode.hpp>         // for INode
#include <kaad/graph/node_handle.hpp>   // for Node
#include <kaad/graph/operator_node.hpp> // for OperatorNode
#include <kaad/operations/slice.hpp>    // for Slice
#include <kaad/static_vector.hpp>       // for StaticVector
#include <kaad/tensor/tensor_types.hpp> // for Shape
#include <memory>                       // for unique_ptr, make_unique
#include <vector>                       // for vector

namespace kaad {

Node slice(Graph &rec, Node input, Shape size,
           StaticVector<std::size_t> start) {

    rec.nodes.push_back(std::make_unique<OperatorNode<operations::Slice>>(
        std::array{rec.get_node(input)}, size, start));

    return rec.back_handle();
}

} // namespace kaad
