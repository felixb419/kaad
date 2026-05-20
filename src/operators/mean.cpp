#include "../graph/operator_node.hpp"
#include "../operations/full_reduce.hpp"
#include "../operations/reduce.hpp"

#include <array>
#include <cstddef>
#include <kaad/exceptions.hpp>
#include <kaad/graph/graph.hpp>
#include <kaad/graph/node_handle.hpp>
#include <kaad/operators/operators.hpp>
#include <memory>
#include <vector>

namespace kaad {

Node mean(Graph &graph, Node input) {

    graph.nodes.push_back(
        std::make_unique<OperatorNode<operations::FullReduceMean>>(
            std::array{graph.get_node(input)}));

    return graph.back_handle();
}

Node mean(Graph &graph, Node input, std::size_t axis, bool keep_rank) {

    try {

        graph.nodes.push_back(
            std::make_unique<OperatorNode<operations::ReduceMean>>(
                std::array{graph.get_node(input)},
                operations::ReduceMean::Metadata{axis, keep_rank}));
    }

    catch (ShapeError &err) {
        // input has rank-1
        return mean(graph, input);
    }

    catch (ArgumentError &err) {
        // axis is not a valid index
        throw ArgumentError(
            make_graph_errmsg(graph.nodes.size(), "mean", err.what()));
    }

    return graph.back_handle();
}

} // namespace kaad
