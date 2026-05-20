#include "../../graph/operator_node.hpp"
#include "../../operations/full_reduce.hpp"
#include "../../operations/reduce.hpp"

#include <array>
#include <cstddef>
#include <kaad/exceptions.hpp>
#include <kaad/graph/graph.hpp>
#include <kaad/graph/node_handle.hpp>
#include <vector>

namespace kaad {

Node Graph::mean(Node input) {

    this->nodes.push_back(new OperatorNode<operations::FullReduceMean>(
        std::array{this->get_node(input)}));

    return Node(this->nodes.size() - 1, this);
}

Node Graph::mean(Node input, std::size_t axis, bool keep_rank) {

    try {

        this->nodes.push_back(new OperatorNode<operations::ReduceMean>(
            std::array{this->get_node(input)},
            operations::ReduceMean::Metadata{axis, keep_rank}));
    }

    catch (ShapeError &err) {
        // input has rank-1
        return this->mean(input);
    }

    catch (ArgumentError &err) {
        // axis is not a valid index
        throw ArgumentError(
            make_graph_errmsg(this->nodes.size(), "mean", err.what()));
    }

    return Node(this->nodes.size() - 1, this);
}

} // namespace kaad
