#include "../../include/kaad/exceptions.hpp"          // for make_graph_e...
#include "../../include/kaad/functions/adjoint.hpp"   // for scalarOut
#include "../../include/kaad/functions/kernels.hpp"   // for Sum
#include "../../include/kaad/functions/primal.hpp"    // for scalarOut
#include "../../include/kaad/graph/graph.hpp"         // for Computation_...
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
#include <utility>                                    // for move
#include <vector>                                     // for vector

namespace kaad {

Node sum(Graph &rec, Node A) {

    INode *A_ptr = rec.get_node(A);
    Tensor &A_val = A_ptr->value();

    using Kernel = struct Kernels::Sum<Scalar>;
    functions::primal::unary::pointwise_fn<Kernel> op =
        functions::primal::unary::scalarOut<Kernel>;
    functions::adjoint::unary::pointwise_fn<Kernel> grad =
        functions::adjoint::unary::scalarOut<Kernel>;

    rec.nodes.push_back(std::make_unique<Node_unary<Kernel>>(
        op, grad, A_ptr, std::array<int, 0>{}));
    static_cast<Node_unary<Kernel> *>(rec.nodes.back().get())->end =
        A_val.data() + A_val.size(); // override end from constructor

    return rec.back_handle();
}

Node sum(Graph &rec, Node A, int dim, bool keep_rank) {
    std::size_t recLen = rec.nodes.size();

    INode *A_ptr = rec.get_node(A);
    Tensor &A_val = A_ptr->value();

    if (dim < 0 || dim >= static_cast<int>(A_val.rank())) {
        throw argument_error(
            make_graph_errmsg("argument error", recLen, "sum",
                              "dim has to be a valid index of A.shape",
                              {{"A.shape", A_val.shape()}}, {{"dim", dim}}));
    }

    if (A_val.rank() == 1) {
        return sum(rec, A);
    }

    std::size_t newLen = A_val.rank();
    std::vector<int> newShape(newLen);
    if (keep_rank) {
        std::copy(A_val.shape().begin(), A_val.shape().end(), newShape.begin());
        newShape[dim] = 1;

    } else {

        newShape.resize(newShape.size() - 1);

        std::copy(A_val.shape().begin(), A_val.shape().begin() + dim,
                  newShape.begin());
        std::copy(A_val.shape().begin() + dim + 1, A_val.shape().end(),
                  newShape.begin() + dim);
    }

    rec.nodes.push_back(std::make_unique<Node_sum_dim>(A_ptr, dim, newShape));
    return rec.back_handle();
}

} // namespace kaad
