#include <kaad/graph/nodes/batch_matmul.hpp>

#include <algorithm>                   // for copy, max
#include <array>                       // for array
#include <kaad/graph/common.hpp>       // for combine_matrix
#include <kaad/graph/dispatchers.hpp>  // for get_batch_ma...
#include <kaad/graph/nodes/inode.hpp>  // for INode
#include <kaad/max_rank.hpp>           // for KAAD_MAX_RANK
#include <kaad/scalar.hpp>             // for Scalar
#include <kaad/tensor/tensor.hpp>      // for Tensor
#include <kaad/tensor/tensor_view.hpp> // for TensorView
#include <utility>                     // for swap
#include <vector>                      // for vector

namespace kaad {

void metadata_impl(TensorViewConst lhs, TensorViewConst rhs,
                   TensorViewConst res, int *&lhs_stride, int *&rhs_stride,
                   int *&res_stride, int *&c_shape_broadcast, int &a_off,
                   int &b_off, int &shared_dim, std::size_t &res_rank) {
    a_off = lhs.stride[lhs.rank() - 1];
    b_off = rhs.stride[rhs.rank() - 2];
    shared_dim = lhs.shape[lhs.rank() - 1];

    res_rank = std::max(lhs.rank(), rhs.rank());
    c_shape_broadcast = new int[res_rank];

    detail::combine_matrix(lhs.shape.data(), lhs.rank(), rhs.shape.data(),
                           rhs.rank(), c_shape_broadcast, res_rank);

    lhs_stride = new int[res_rank];
    rhs_stride = new int[res_rank];
    res_stride = new int[res_rank];

    int idx;
    int idxA;
    int idxB;
    int idxC;
    int rank = static_cast<int>(res_rank);
    for (int i = 1; i <= rank; i++) {
        idx = rank - i;
        idxA = static_cast<int>(lhs.rank()) - i;
        lhs_stride[idx] = idxA >= 0 ? lhs.stride[idxA] : 0;
        idxB = static_cast<int>(rhs.rank()) - i;
        rhs_stride[idx] = idxB >= 0 ? rhs.stride[idxB] : 0;
        idxC = static_cast<int>(res.rank()) - i;
        res_stride[idx] = idxC >= 0 ? res.stride[idxC] : 0;
    }

    lhs_stride[res_rank - 1] = 0;
    rhs_stride[res_rank - 2] = 0;
}

void Node_batch_matmul::metadata() {
    // compute metadata
    TensorViewConst lhs = this->lhs->value().view();
    TensorViewConst rhs = this->rhs->value().view();
    TensorViewConst value = this->value().view();

    std::vector<int> a_T_shape(lhs.rank());
    std::vector<int> a_T_stride(lhs.rank());

    std::ranges::copy(lhs.shape, a_T_shape.data());
    std::swap(a_T_shape[lhs.rank() - 1], a_T_shape[lhs.rank() - 2]);

    std::ranges::copy(lhs.stride, a_T_stride.data());
    std::swap(a_T_stride[lhs.rank() - 1], a_T_stride[lhs.rank() - 2]);

    TensorViewConst a_T = lhs;
    a_T.shape = std::span<const int>(a_T_shape);
    a_T.stride = std::span<const int>(a_T_stride);

    std::vector<int> b_T_shape(rhs.rank());
    std::vector<int> b_T_stride(rhs.rank());

    std::ranges::copy(rhs.shape, b_T_shape.data());
    std::swap(b_T_shape[rhs.rank() - 1], b_T_shape[rhs.rank() - 2]);

    std::ranges::copy(rhs.stride, b_T_stride.data());
    std::swap(b_T_stride[rhs.rank() - 1], b_T_stride[rhs.rank() - 2]);

    TensorViewConst b_T = rhs;
    b_T.shape = std::span<const int>(b_T_shape);
    b_T.stride = std::span<const int>(b_T_stride);

    metadata_impl(lhs, rhs, value, this->lhs_stride[0], this->rhs_stride[0],
                  this->value_stride[0], this->value_shape_broadcast[0],
                  this->lhs_colStride[0], this->rhs_rowStride[0],
                  this->shared_dim[0], this->value_rank);
    // NOLINTNEXTLINE(readability-suspicious-call-argument)
    metadata_impl(value, b_T, lhs, this->value_stride[1], this->rhs_stride[1],
                  this->lhs_stride[1], this->value_shape_broadcast[1],
                  this->lhs_colStride[1], this->rhs_rowStride[1],
                  this->shared_dim[1], this->value_rank);
    // NOLINTNEXTLINE(readability-suspicious-call-argument)
    metadata_impl(a_T, value, rhs, this->lhs_stride[2], this->value_stride[2],
                  this->rhs_stride[2], this->value_shape_broadcast[2],
                  this->lhs_colStride[2], this->rhs_rowStride[2],
                  this->shared_dim[2], this->value_rank);

    // assign compile-time recursive function
    if (this->value_rank <= KAAD_MAX_RANK) {
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
                this->gradient().data(), lhs_stride.begin() + 1,
                rhs_stride.begin() + 1, value_stride.begin() + 1,
                value_shape_broadcast.begin() + 1, lhs_colStride.begin() + 1,
                rhs_rowStride.begin() + 1, shared_dim.begin() + 1, value_rank);

    if (!this->lhs->isInput()) {
        this->lhs->getGrad();
    }
    if (!this->rhs->isInput()) {
        this->rhs->getGrad();
    }
}

} // namespace kaad
