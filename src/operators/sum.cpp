#include "../../include/kaad/exceptions.hpp"          // for argument_error
#include "../../include/kaad/functions/adjoint.hpp"   // for pointwise_fn
#include "../../include/kaad/functions/kernels.hpp"   // for Sum
#include "../../include/kaad/functions/primal.hpp"    // for pointwise_fn
#include "../../include/kaad/graph/graph.hpp"         // for Graph, sum
#include "../../include/kaad/graph/node_handle.hpp"   // for Node
#include "../../include/kaad/graph/nodes/inode.hpp"   // for INode
#include "../../include/kaad/graph/nodes/sum_dim.hpp" // for Node_sum_dim
#include "../../include/kaad/graph/nodes/unary.hpp"   // for Node_unary
#include "../../include/kaad/operators/operators.hpp" // for sum
#include "../../include/kaad/scalar.hpp"              // for Scalar
#include "../../include/kaad/tensor/tensor.hpp"       // for Tensor
#include <algorithm>                                  // for copy
#include <array>                                      // for array
#include <cstddef>                                    // for size_t
#include <memory>                                     // for unique_ptr
#include <span>                                       // for span
#include <string>                                     // for basic_string
#include <utility>                                    // for cmp_greater_equal
#include <vector>                                     // for vector

namespace kaad {

Node sum(Graph &rec, Node input) {

    INode *input_ptr = rec.get_node(input);
    Tensor &input_val = input_ptr->value();

    using Kernel = struct Kernels::Sum<Scalar>;
    functions::primal::unary::pointwise_fn<Kernel> func =
        functions::primal::unary::scalarOut<Kernel>;
    functions::adjoint::unary::pointwise_fn<Kernel> grad =
        functions::adjoint::unary::scalarOut<Kernel>;

    rec.nodes.push_back(std::make_unique<Node_unary<Kernel>>(
        func, grad, input_ptr, std::array<int, 0>{}));
    static_cast<Node_unary<Kernel> *>(rec.nodes.back().get())->end =
        input_val.data() + input_val.size(); // override end from constructor

    return rec.back_handle();
}

Node sum(Graph &rec, Node input, int dim, bool keep_rank) {
    std::size_t recLen = rec.nodes.size();

    INode *input_ptr = rec.get_node(input);
    Tensor &input_val = input_ptr->value();

    if (dim < 0 || std::cmp_greater_equal(dim, input_val.rank())) {
        throw argument_error(make_graph_errmsg(
            "argument error", recLen, "sum",
            "dim has to be a valid index of A.shape",
            {{"A.shape", input_val.shape()}}, {{"dim", dim}}));
    }

    if (input_val.rank() == 1) {
        return sum(rec, input);
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
        std::make_unique<Node_sum_dim>(input_ptr, dim, newShape));
    return rec.back_handle();
}

} // namespace kaad
