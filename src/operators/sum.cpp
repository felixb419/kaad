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
                std::array{rec.get_node(input)},
                operations::ReduceSum::Metadata{axis, keep_rank}));
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
