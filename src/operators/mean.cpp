#include <kaad/operators/operators.hpp> // for mean

#include "../exceptions.hpp"           // for ArgumentError, make_graph_...
#include "../graph/nodes/mean.hpp"     // for NodeMean
#include "../graph/nodes/mean_dim.hpp" // for NodeMeanDim
#include <algorithm>                   // for copy
#include <cstddef>                     // for size_t
#include <kaad/graph/graph.hpp>        // for Graph, mean
#include <kaad/graph/node_handle.hpp>  // for Node
#include <kaad/graph/nodes/inode.hpp>  // for INode
#include <kaad/tensor/tensor.hpp>      // for Tensor
#include <memory>                      // for unique_ptr, make_unique
#include <span>                        // for span
#include <string>                      // for basic_string
#include <utility>                     // for cmp_greater_equal, pair
#include <vector>                      // for vector

namespace kaad {

Node mean(Graph &rec, Node input) {

    rec.nodes.push_back(std::make_unique<NodeMean>(rec.get_node(input)));
    return rec.back_handle();
}

Node mean(Graph &rec, Node input, int dim, bool keep_rank) {
    std::size_t rec_len = rec.nodes.size();

    INode *input_ptr = rec.get_node(input);
    Tensor &input_val = input_ptr->value();

    if (dim < 0 || std::cmp_greater_equal(dim, input_val.rank())) {
        throw ArgumentError(make_graph_errmsg(
            "argument error", rec_len, "mean",
            "dim has to be a valid index of A.shape",
            {{"A.shape", input_val.shape()}}, {{"dim", dim}}));
    }

    if (input_val.rank() == 1) {
        return mean(rec, input);
    }

    std::size_t new_len = input_val.rank();
    std::vector<int> new_shape(new_len);
    if (keep_rank) {
        std::copy(input_val.shape().begin(), input_val.shape().end(),
                  new_shape.begin());
        new_shape[dim] = 1;

    } else {

        new_shape.resize(new_shape.size() - 1);

        std::copy(input_val.shape().begin(), input_val.shape().begin() + dim,
                  new_shape.begin());
        std::copy(input_val.shape().begin() + dim + 1, input_val.shape().end(),
                  new_shape.begin() + dim);
    }

    rec.nodes.push_back(
        std::make_unique<NodeMeanDim>(input_ptr, dim, new_shape));
    return rec.back_handle();
}

} // namespace kaad
