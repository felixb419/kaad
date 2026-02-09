#include "operators.hpp"

#include "../../tensor/tensor.hpp"  // for Tensor
#include "../computation_graph.hpp" // for Computation_graph
#include "../node_handle.hpp"       // for Node_handle
#include "../nodes/outer.hpp"       // for Node_outer
#include <memory>                   // for std::make_unique

namespace kaad {

Node_handle outer(Computation_graph &rec, Node_handle A, Node_handle B) {
    int recLen = rec.nodes.size();

    INode *A_ptr = rec.get_node(A);
    INode *B_ptr = rec.get_node(B);
    Tensor &A_val = A_ptr->value;
    Tensor &B_val = B_ptr->value;

    size_t newLen = A_val.nDims() + B_val.nDims();
    std::vector<int> newShape(newLen);
    std::copy(A_val.shape_begin(), A_val.shape_end(), newShape.begin());
    std::copy(B_val.shape_begin(), B_val.shape_end(),
              newShape.begin() + A_val.nDims());

    rec.nodes.push_back(
        std::move(std::make_unique<Node_outer>(A_ptr, B_ptr, newShape)));

    return rec.back_handle();
}

} // namespace kaad
