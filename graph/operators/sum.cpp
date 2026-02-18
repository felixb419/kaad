#include "operators.hpp"

#include "../../tensor/tensor.hpp"           // for Tensor
#include "../../tensorfuncs/adjoint_ops.hpp" // for tensorfuncs::adjoint
#include "../../tensorfuncs/kernels.hpp"     // for Kernels::Sum
#include "../computation_graph.hpp"          // for Computation_graph
#include "../node_handle.hpp"                // for Node_handle
#include "../nodes/inode.hpp"                // for INode
#include "../nodes/sum_dim.hpp"              // for Node_sum_dim
#include "../nodes/unary.hpp"                // for Node_unary
#include "exceptions.hpp"                    // for argument_error
#include <memory>                            // for std::make_unique

namespace kaad {

Node_handle sum(Computation_graph &rec, Node_handle A) {
    int recLen = rec.nodes.size();

    INode *A_ptr = rec.get_node(A);
    Tensor &A_val = A_ptr->value;

    using Kernel = class Kernels::Sum<Scalar>;
    tensorfuncs::primal::unary::pointwise_fn<Kernel> op =
        tensorfuncs::primal::unary::scalarOut<Kernel>;
    tensorfuncs::adjoint::unary::pointwise_fn<Kernel> grad =
        tensorfuncs::adjoint::unary::scalarOut<Kernel>;

    rec.nodes.push_back(std::move(
        std::make_unique<Node_unary<Kernel>>(op, grad, A_ptr, std::array{1})));
    static_cast<Node_unary<Kernel> *>(rec.nodes.back().get())->end =
        A_val.data() + A_val.size(); // override end from constructor

    return rec.back_handle();
}

Node_handle sum(Computation_graph &rec, Node_handle A, int dim,
                bool keepNDims) {
    int recLen = rec.nodes.size();

    INode *A_ptr = rec.get_node(A);
    Tensor &A_val = A_ptr->value;

    if (dim < 0 || dim >= A_val.rank()) {
        throw argument_error(recLen, "sum",
                             "dim has to be a valid index of A.shape",
                             {{"A.shape", A_val.shape()}}, {{"dim", dim}});
    }

    if (A_val.rank() == 1) {
        return sum(rec, A);
    }

    std::size_t newLen = A_val.rank();
    std::vector<int> newShape(newLen);
    if (keepNDims) {
        std::copy(A_val.shape().begin(), A_val.shape().end(), newShape.begin());
        newShape[dim] = 1;

    } else {
        newLen--;
        std::copy(A_val.shape().begin(), A_val.shape().begin() + dim,
                  newShape.begin());
        std::copy(A_val.shape().begin() + dim + 1, A_val.shape().end(),
                  newShape.begin() + dim);
    }

    rec.nodes.push_back(
        std::move(std::make_unique<Node_sum_dim>(A_ptr, dim, newShape)));
    return rec.back_handle();
}

} // namespace kaad
