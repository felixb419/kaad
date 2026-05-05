#include <kaad/operators/operators.hpp> // for mean

#include <array>                          // for array
#include <cstddef>                        // for size_t
#include <kaad/exceptions.hpp>            // for ArgumentError, ShapeError
#include <kaad/functions/full_reduce.hpp> // for FullReduceMean
#include <kaad/functions/reduce.hpp>      // for ReduceMean
#include <kaad/graph/graph.hpp>           // for Graph, mean
#include <kaad/graph/inode.hpp>           // for INode
#include <kaad/graph/node_handle.hpp>     // for Node
#include <kaad/graph/operator_node.hpp>   // for OperatorNode
#include <memory>                         // for unique_ptr, make_unique
#include <string>                         // for basic_string
#include <vector>                         // for vector

namespace kaad {

Node mean(Graph &rec, Node input) {

    rec.nodes.push_back(
        std::make_unique<OperatorNode<functions::FullReduceMean>>(
            std::array{rec.get_node(input)}));

    return rec.back_handle();
}

Node mean(Graph &rec, Node input, std::size_t dim, bool keep_rank) {

    try {

        rec.nodes.push_back(
            std::make_unique<OperatorNode<functions::ReduceMean>>(
                std::array{rec.get_node(input)}, dim, keep_rank));
    }

    catch (ShapeError &err) {
        // input has rank-1
        return mean(rec, input);
    }

    catch (ArgumentError &err) {
        // dim is not a valid index
        throw ArgumentError(
            make_graph_errmsg(rec.nodes.size(), "mean", err.what()));
    }

    return rec.back_handle();
}

} // namespace kaad
