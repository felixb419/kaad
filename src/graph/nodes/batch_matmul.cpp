#include "../../../include/kaad/graph/nodes/batch_matmul.hpp"
#include "../../../include/kaad/functions/adjoint.hpp"  // for batch_matmu...
#include "../../../include/kaad/functions/primal.hpp"   // for batch_matmu...
#include "../../../include/kaad/graph/common.hpp"       // for combine_matrix
#include "../../../include/kaad/graph/dispatchers.hpp"  // for get_batch_m...
#include "../../../include/kaad/scalar.hpp"             // for Scalar
#include "../../../include/kaad/tensor/tensor.hpp"      // for Tensor
#include "../../../include/kaad/tensor/tensor_view.hpp" // for Tensor_view
#include <algorithm>                                    // for copy, max
#include <array>                                        // for array
#include <utility>                                      // for swap
#include <vector>                                       // for vector

namespace kaad {

void metadata_impl(Tensor_view lhs, Tensor_view rhs, Tensor_view res,
                   int *&lhs_stride, int *&rhs_stride, int *&res_stride,
                   int *&c_shape_broadcast, int &a_off, int &b_off, int &k,
                   std::size_t &D) {
    a_off = lhs.stride[lhs.rank - 1];
    b_off = rhs.stride[rhs.rank - 2];
    k = lhs.shape[lhs.rank - 1];

    D = std::max(lhs.rank, rhs.rank);
    c_shape_broadcast = new int[D];

    detail::combine_matrix(lhs.shape, lhs.rank, rhs.shape, rhs.rank,
                           c_shape_broadcast, D);

    lhs_stride = new int[D];
    rhs_stride = new int[D];
    res_stride = new int[D];

    int idx, idxA, idxB, idxC;
    for (size_t i = 1; i <= D; i++) {
        idx = D - i;
        idxA = lhs.rank - i;
        lhs_stride[idx] = idxA >= 0 ? lhs.stride[idxA] : 0;
        idxB = rhs.rank - i;
        rhs_stride[idx] = idxB >= 0 ? rhs.stride[idxB] : 0;
        idxC = res.rank - i;
        res_stride[idx] = idxC >= 0 ? res.stride[idxC] : 0;
    }

    lhs_stride[D - 1] = 0;
    rhs_stride[D - 2] = 0;
}

void Node_batch_matmul::metadata() {
    // compute metadata
    Tensor_view lhs = this->lhs->value().view();
    Tensor_view rhs = this->rhs->value().view();
    Tensor_view value = this->value().view();

    std::vector<int> a_T_shape(lhs.rank);
    std::vector<int> a_T_stride(lhs.rank);

    std::copy(lhs.shape, lhs.shape + lhs.rank, a_T_shape.data());
    std::swap(a_T_shape[lhs.rank - 1], a_T_shape[lhs.rank - 2]);
    std::copy(lhs.stride, lhs.stride + lhs.rank, a_T_stride.data());
    std::swap(a_T_stride[lhs.rank - 1], a_T_stride[lhs.rank - 2]);

    Tensor_view a_T = lhs;
    a_T.shape = a_T_shape.data();
    a_T.stride = a_T_stride.data();

    std::vector<int> b_T_shape(rhs.rank);
    std::vector<int> b_T_stride(rhs.rank);

    std::copy(rhs.shape, rhs.shape + rhs.rank, b_T_shape.data());
    std::swap(b_T_shape[rhs.rank - 1], b_T_shape[rhs.rank - 2]);
    std::copy(rhs.stride, rhs.stride + rhs.rank, b_T_stride.data());
    std::swap(b_T_stride[rhs.rank - 1], b_T_stride[rhs.rank - 2]);

    Tensor_view b_T = rhs;
    b_T.shape = b_T_shape.data();
    b_T.stride = b_T_stride.data();

    metadata_impl(lhs, rhs, value, this->lhs_stride[0], this->rhs_stride[0],
                  this->value_stride[0], this->value_shape_broadcast[0],
                  this->lhs_colStride[0], this->rhs_rowStride[0],
                  this->shared_dim[0], this->value_rank);
    metadata_impl(value, b_T, lhs, this->value_stride[1], this->rhs_stride[1],
                  this->lhs_stride[1], this->value_shape_broadcast[1],
                  this->lhs_colStride[1], this->rhs_rowStride[1],
                  this->shared_dim[1], this->value_rank);
    metadata_impl(a_T, value, rhs, this->lhs_stride[2], this->value_stride[2],
                  this->rhs_stride[2], this->value_shape_broadcast[2],
                  this->lhs_colStride[2], this->rhs_rowStride[2],
                  this->shared_dim[2], this->value_rank);

    // assign compile-time recursive function
    if (this->value_rank <= Dispatchers::MAX_NDIMS) {
        this->forward_op =
            Dispatchers::get_batch_matmul<Scalar>()[this->value_rank];
        this->backward_op =
            Dispatchers::get_batch_matmul_grad<Scalar>()[this->value_rank];
    }
}

Node_batch_matmul::Node_batch_matmul(INode *lhs_ptr, INode *rhs_ptr,
                                     std::span<const int> value_shape)
    : INode(value_shape, false), lhs(lhs_ptr), rhs(rhs_ptr) {

    this->metadata();
}

Node_batch_matmul::~Node_batch_matmul() noexcept {
    for (int i = 0; i < 3; i++) {
        delete[] lhs_stride[i];
        delete[] rhs_stride[i];
        delete[] value_stride[i];
        delete[] value_shape_broadcast[i];
    }
}

const char *Node_batch_matmul::node_type() const noexcept {
    return "Node_batch_matmul";
}

void Node_batch_matmul::eval() {
    if (!this->evaluated()) {
        this->lhs->eval();
        this->rhs->eval();

        forward_op(this->lhs->value().data(), this->rhs->value().data(),
                   this->value().data(), lhs_stride[0], rhs_stride[0],
                   value_stride[0], value_shape_broadcast[0], lhs_colStride[0],
                   rhs_rowStride[0], shared_dim[0], value_rank);
        this->evaluated_ = true;
    }
}

void Node_batch_matmul::getGrad() {
    backward_op(this->lhs->value().data(), this->lhs->gradient().data(),
                this->rhs->value().data(), this->rhs->gradient().data(),
                this->gradient().data(), lhs_stride + 1, rhs_stride + 1,
                value_stride + 1, value_shape_broadcast + 1, lhs_colStride + 1,
                rhs_rowStride + 1, shared_dim + 1, value_rank);

    if (this->lhs->hasInputs()) {
        this->lhs->getGrad();
    }
    if (this->rhs->hasInputs()) {
        this->rhs->getGrad();
    }
}

} // namespace kaad
