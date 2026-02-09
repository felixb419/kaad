#include "operators.hpp"

#include "../../scalar.hpp"                  // for Scalar
#include "../../tensor/tensor.hpp"           // for Tensor
#include "../../tensorfuncs/adjoint_ops.hpp" // for tensorfuncs::adjoint
#include "../../tensorfuncs/kernels.hpp"     // for Kernels
#include "../../tensorfuncs/primal_ops.hpp"  // for tensorfuncs::primal
#include "../computation_graph.hpp"          // for Computation_graph
#include "../node_handle.hpp"                // for Node_handle
#include "../nodes/binary.hpp"               // for Node_binary
#include "exceptions.hpp"                    // for shape_error
#include <memory>                            // for std::make_unique

namespace kaad {

Node_handle dot(Computation_graph &rec, Node_handle A, Node_handle B) {
    tensorfuncs::primal::binary::pointwise_fn<Scalar, Kernels::Null> scalar =
        tensorfuncs::primal::binary::scalarDot<Scalar>;
    tensorfuncs::adjoint::binary::pointwise_fn<Scalar, Kernels::Null>
        scalar_grad =
            tensorfuncs::adjoint::binary::scalarDot<Scalar, Kernels::Null>;

    tensorfuncs::primal::binary::pointwise_fn<Scalar, Kernels::Null> dot =
        tensorfuncs::primal::binary::dot<Scalar, Kernels::Null>;
    tensorfuncs::adjoint::binary::pointwise_fn<Scalar, Kernels::Null> dot_grad =
        tensorfuncs::adjoint::binary::dot<Scalar, Kernels::Null>;

    int recLen = rec.nodes.size();

    INode *A_ptr = rec.get_node(A);
    INode *B_ptr = rec.get_node(B);
    Tensor &A_val = A_ptr->value;
    Tensor &B_val = B_ptr->value;

    bool A_scalar = A_val.nDims() == 1 && A_val.shape()[0] == 1;
    bool B_scalar = B_val.nDims() == 1 && B_val.shape()[0] == 1;
    if (B_scalar) {

        rec.nodes.push_back(
            std::move(std::make_unique<Node_binary<Kernels::Null>>(
                scalar, scalar_grad, A_ptr, B_ptr, ((Scalar)0))));
        static_cast<Node_binary<Kernels::Null> *>(rec.nodes.back().get())->end =
            A_val.data() + A_val.size();

    } else if (A_scalar) {

        rec.nodes.push_back(
            std::move(std::make_unique<Node_binary<Kernels::Null>>(
                scalar, scalar_grad, B_ptr, A_ptr, ((Scalar)0))));
        static_cast<Node_binary<Kernels::Null> *>(rec.nodes.back().get())->end =
            B_val.data() + B_val.size(); // override end from constructor

    } else if (A_val.nDims() == 1 && B_val.nDims() == 1 &&
               std::equal(A_val.shape_begin(), A_val.shape_end(),
                          B_val.shape_begin())) {

        rec.nodes.push_back(
            std::move(std::make_unique<Node_binary<Kernels::Null>>(
                dot, dot_grad, A_ptr, B_ptr, ((Scalar)0))));
        static_cast<Node_binary<Kernels::Null> *>(rec.nodes.back().get())->end =
            A_val.data() + A_val.size();

    } else {
        throw shape_error(
            recLen, "dot", "incompatible tensor shapes for dot product",
            {{"A.shape", A_val.shape()}, {"B.shape", B_val.shape()}});
    }

    return rec.back_handle();
}

} // namespace kaad
