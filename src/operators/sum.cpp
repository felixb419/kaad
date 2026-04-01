#include <kaad/operators/operators.hpp> // for sum

#include "../graph/nodes/sum_dim.hpp"   // for NodeSumDim
#include "../graph/nodes/unary.hpp"     // for NodeUnary
#include <algorithm>                    // for copy
#include <cstddef>                      // for size_t
#include <kaad/exceptions.hpp>          // for ArgumentError, make_graph_er...
#include <kaad/functions/adjoint.hpp>   // for pointwise_fn, scalar_out
#include <kaad/functions/kernels.hpp>   // for Sum
#include <kaad/functions/primal.hpp>    // for pointwise_fn, scalar_out
#include <kaad/graph/graph.hpp>         // for Graph, sum
#include <kaad/graph/node_handle.hpp>   // for Node
#include <kaad/graph/nodes/inode.hpp>   // for INode
#include <kaad/scalar.hpp>              // for Scalar
#include <kaad/tensor/tensor.hpp>       // for Tensor
#include <kaad/tensor/tensor_types.hpp> // for Shape
#include <memory>                       // for unique_ptr, allocator, make_...
#include <string>                       // for char_traits, basic_string
#include <utility>                      // for cmp_greater_equal
#include <vector>                       // for vector

namespace kaad {

Node sum(Graph &rec, Node input) {

    INode *input_ptr = rec.get_node(input);
    Tensor &input_val = input_ptr->value();

    using Kernel = struct Kernels::Sum<Scalar>;
    functions::primal::unary::pointwise_fn<Kernel> func =
        functions::primal::unary::scalar_out<Kernel>;
    functions::adjoint::unary::pointwise_fn<Kernel> grad =
        functions::adjoint::unary::scalar_out<Kernel>;

    rec.nodes.push_back(std::make_unique<NodeUnary<Kernel>>(
        func, grad, input_ptr, SCALAR_SHAPE));
    static_cast<NodeUnary<Kernel> *>(rec.nodes.back().get())->end =
        input_val.data() + input_val.size(); // override end from constructor

    return rec.back_handle();
}

Node sum(Graph &rec, Node input, int dim, bool keep_rank) {
    std::size_t rec_len = rec.nodes.size();

    INode *input_ptr = rec.get_node(input);
    Tensor &input_val = input_ptr->value();

    if (dim < 0 || std::cmp_greater_equal(dim, input_val.rank())) {
        throw ArgumentError(make_graph_errmsg(
            rec_len, "sum",
            "dim has to be a valid index of A.shape, A.shape=" +
                to_string(input_val.shape()) + ", dim=" + std::to_string(dim)));
    }

    if (input_val.rank() == 1) {
        return sum(rec, input);
    }

    std::size_t new_len = input_val.rank();
    Shape new_shape(new_len);
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
        std::make_unique<NodeSumDim>(input_ptr, dim, new_shape));
    return rec.back_handle();
}

} // namespace kaad
