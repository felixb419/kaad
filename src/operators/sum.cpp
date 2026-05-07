#include <kaad/operators/operators.hpp> // for sum

#include <array>                           // for array
#include <cstddef>                         // for size_t
#include <kaad/exceptions.hpp>             // for ArgumentError, ShapeError
#include <kaad/graph/graph.hpp>            // for Graph, sum
#include <kaad/graph/inode.hpp>            // for INode
#include <kaad/graph/node_handle.hpp>      // for Node
#include <kaad/graph/operator_node.hpp>    // for OperatorNode
#include <kaad/operations/full_reduce.hpp> // for FullReduceSum
#include <kaad/operations/reduce.hpp>      // for ReduceSum
#include <memory>                          // for unique_ptr, make_unique
#include <string>                          // for basic_string
#include <vector>                          // for vector

namespace kaad {

Node sum(Graph &rec, Node input) {

    rec.nodes.push_back(
        std::make_unique<OperatorNode<operations::FullReduceSum>>(
            std::array{rec.get_node(input)}));

    return rec.back_handle();
}

Node sum(Graph &rec, Node input, std::size_t axis, bool keep_rank) {

    try {

        rec.nodes.push_back(
            std::make_unique<OperatorNode<operations::ReduceSum>>(
                std::array{rec.get_node(input)}, axis, keep_rank));
    }

    catch (ShapeError &err) {
        // input has rank-1
        return sum(rec, input);
    }

    catch (ArgumentError &err) {
        // axis is not a valid index
        throw ArgumentError(
            make_graph_errmsg(rec.nodes.size(), "sum", err.what()));
    }

    return rec.back_handle();
}

} // namespace kaad
