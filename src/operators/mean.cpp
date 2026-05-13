#include <kaad/operators/operators.hpp>

#include "../graph/operator_node.hpp"
#include "../operations/full_reduce.hpp"
#include "../operations/reduce.hpp"
#include <array>
#include <cstddef>
#include <kaad/exceptions.hpp>
#include <kaad/graph/graph.hpp>
#include <kaad/graph/node_handle.hpp>
#include <memory>
#include <vector>

namespace kaad {

Node mean(Graph &rec, Node input) {

    rec.nodes.push_back(
        std::make_unique<OperatorNode<operations::FullReduceMean>>(
            std::array{rec.get_node(input)}));

    return rec.back_handle();
}

Node mean(Graph &rec, Node input, std::size_t axis, bool keep_rank) {

    try {

        rec.nodes.push_back(
            std::make_unique<OperatorNode<operations::ReduceMean>>(
                std::array{rec.get_node(input)}, axis, keep_rank));
    }

    catch (ShapeError &err) {
        // input has rank-1
        return mean(rec, input);
    }

    catch (ArgumentError &err) {
        // axis is not a valid index
        throw ArgumentError(
            make_graph_errmsg(rec.nodes.size(), "mean", err.what()));
    }

    return rec.back_handle();
}

} // namespace kaad
