#include "operators.hpp"

#include "../../tensor/tensor.hpp"  // for Tensor
#include "../computation_graph.hpp" // for Computation_graph
#include "../node_handle.hpp"       // for Node_handle
#include "../nodes/mean.hpp"        // for Node_mean
#include "../nodes/mean_dim.hpp"    // for Node_mean_dim
#include "exceptions.hpp"           // for param_error
#include <memory>                   // for std::make_unique

namespace kaad {

Node_handle mean(Computation_graph &rec, Node_handle A) {
    int recLen = rec.nodes.size();

    rec.nodes.push_back(
        std::move(std::make_unique<Node_mean>(rec.get_node(A), (Scalar)0)));
    return rec.back_handle();
}

Node_handle mean(Computation_graph &rec, Node_handle A, int dim,
                 bool keepNDims) {
    int recLen = rec.nodes.size();

    INode *A_ptr = rec.get_node(A);
    Tensor &A_val = A_ptr->value;

    if (dim < 0 || dim >= A_val.nDims()) {
        throw argument_error(recLen, "mean",
                             "dim has to be a valid index of A.shape",
                             {{"A.shape", A_val.shape()}}, {{"dim", dim}});
    }

    if (A_val.nDims() == 1) {
        return mean(rec, A);
    }

    size_t newLen = A_val.nDims();
    std::vector<int> newShape(newLen);
    if (keepNDims) {
        std::copy(A_val.shape_begin(), A_val.shape_end(), newShape.begin());
        newShape[dim] = 1;

    } else {
        newLen--;
        std::copy(A_val.shape_begin(), A_val.shape_begin() + dim,
                  newShape.begin());
        std::copy(A_val.shape_begin() + dim + 1, A_val.shape_end(),
                  newShape.begin() + dim);
    }

    rec.nodes.push_back(std::move(
        std::make_unique<Node_mean_dim>(A_ptr, dim, newShape, newLen)));
    return rec.back_handle();
}

} // namespace kaad
