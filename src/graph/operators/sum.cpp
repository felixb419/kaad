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

Node Graph::sum(Node input) {

    this->nodes.push_back(new OperatorNode<operations::FullReduceSum>(
        std::array{this->get_node(input)}));

    return Node(this->nodes.size() - 1, this);
}

Node Graph::sum(Node input, std::size_t axis, bool keep_rank) {

    try {

        this->nodes.push_back(new OperatorNode<operations::ReduceSum>(
            std::array{this->get_node(input)},
            operations::ReduceSum::Metadata{axis, keep_rank}));
    }

    catch (ShapeError &err) {
        // input has rank-1
        return sum(input);
    }

    catch (ArgumentError &err) {
        // axis is not a valid index
        throw ArgumentError(
            make_graph_errmsg(this->nodes.size(), "sum", err.what()));
    }

    return Node(this->nodes.size() - 1, this);
}

} // namespace kaad
