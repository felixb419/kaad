#include "../../include/kaad/graph/nodes/mean.hpp"     // for Node_mean
#include "../../include/kaad/exceptions.hpp"           // for make_graph_errmsg
#include "../../include/kaad/graph/graph.hpp"          // for Graph
#include "../../include/kaad/graph/node_handle.hpp"    // for Node
#include "../../include/kaad/graph/nodes/inode.hpp"    // for INode
#include "../../include/kaad/graph/nodes/mean_dim.hpp" // for Node_mean_dim
#include "../../include/kaad/operators/operators.hpp"  // for mean
#include "../../include/kaad/tensor/tensor.hpp"        // for Tensor
#include <algorithm>                                   // for copy
#include <array>                                       // for array
#include <cstddef>                                     // for size_t
#include <memory>                                      // for unique_ptr, __u...
#include <utility>                                     // for move
#include <vector>                                      // for vector

namespace kaad {

Node mean(Graph &rec, Node input) {

    rec.nodes.push_back(std::make_unique<Node_mean>(rec.get_node(input)));
    return rec.back_handle();
}

Node mean(Graph &rec, Node input, int dim, bool keep_rank) {
    std::size_t recLen = rec.nodes.size();

    INode *input_ptr = rec.get_node(input);
    Tensor &input_val = input_ptr->value();

    if (dim < 0 || dim >= static_cast<int>(input_val.rank())) {
        throw argument_error(make_graph_errmsg(
            "argument error", recLen, "mean",
            "dim has to be a valid index of A.shape",
            {{"A.shape", input_val.shape()}}, {{"dim", dim}}));
    }

    if (input_val.rank() == 1) {
        return mean(rec, input);
    }

    std::size_t newLen = input_val.rank();
    std::vector<int> newShape(newLen);
    if (keep_rank) {
        std::copy(input_val.shape().begin(), input_val.shape().end(),
                  newShape.begin());
        newShape[dim] = 1;

    } else {

        newShape.resize(newShape.size() - 1);

        std::copy(input_val.shape().begin(), input_val.shape().begin() + dim,
                  newShape.begin());
        std::copy(input_val.shape().begin() + dim + 1, input_val.shape().end(),
                  newShape.begin() + dim);
    }

    rec.nodes.push_back(
        std::make_unique<Node_mean_dim>(input_ptr, dim, newShape));
    return rec.back_handle();
}

} // namespace kaad
