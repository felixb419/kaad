#include "../graph/operator_node.hpp"
#include "../operations/full_reduce.hpp"
#include "../operations/reduce.hpp"

#include <array>
#include <cstddef>
#include <kaad/exceptions.hpp>
#include <kaad/graph/graph.hpp>
#include <kaad/graph/node_handle.hpp>
#include <kaad/operators/operators.hpp>
#include <vector>

namespace kaad {

Node sum(Graph &graph, Node input) {

    graph.nodes.push_back(new OperatorNode<operations::FullReduceSum>(
        std::array{graph.get_node(input)}));

    return graph.back_handle();
}

Node sum(Graph &graph, Node input, std::size_t axis, bool keep_rank) {

    try {

        graph.nodes.push_back(new OperatorNode<operations::ReduceSum>(
            std::array{graph.get_node(input)},
            operations::ReduceSum::Metadata{axis, keep_rank}));
    }

    catch (ShapeError &err) {
        // input has rank-1
        return sum(graph, input);
    }

    catch (ArgumentError &err) {
        // axis is not a valid index
        throw ArgumentError(
            make_graph_errmsg(graph.nodes.size(), "sum", err.what()));
    }

    return graph.back_handle();
}

} // namespace kaad
