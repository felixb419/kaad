#include "batch_matmul.hpp"

#include "../../tensor/tensor_view.hpp" // for Tensor_view
#include "../common.hpp"                // for combine_matrix
#include "../dispatchers.hpp" // for get_batch_matmul, get_batch_matmul_grad

namespace kaad {

void metadata_impl(Tensor_view A, Tensor_view B, Tensor_view C, int *&strideA,
                   int *&strideB, int *&strideC, int *&c_shape, int &a_off,
                   int &b_off, int &k, size_t &D) {
    a_off = A.stride[A.nDims - 1];
    b_off = B.stride[B.nDims - 2];
    k = A.shape[A.nDims - 1];

    D = std::max(A.nDims, B.nDims);
    c_shape = new int[D];

    detail::combine_matrix(A.shape, A.nDims, B.shape, B.nDims, c_shape, D);

    strideA = new int[D];
    strideB = new int[D];
    strideC = new int[D];

    int idx, idxA, idxB, idxC;
    for (int i = 1; i <= D; i++) {
        idx = D - i;
        idxA = A.nDims - i;
        strideA[idx] = idxA >= 0 ? A.stride[idxA] : 0;
        idxB = B.nDims - i;
        strideB[idx] = idxB >= 0 ? B.stride[idxB] : 0;
        idxC = C.nDims - i;
        strideC[idx] = idxC >= 0 ? C.stride[idxC] : 0;
    }

    strideA[D - 1] = 0;
    strideB[D - 2] = 0;
}

void Node_batch_matmul::metadata() {
    // compute metadata
    Tensor_view A = this->A->value.view();
    Tensor_view B = this->B->value.view();
    Tensor_view C = this->value.view();

    std::vector<int> a_T_shape(A.nDims);
    std::vector<int> a_T_stride(A.nDims);

    std::copy(A.shape, A.shape + A.nDims, a_T_shape.data());
    std::swap(a_T_shape[A.nDims - 1], a_T_shape[A.nDims - 2]);
    std::copy(A.stride, A.stride + A.nDims, a_T_stride.data());
    std::swap(a_T_stride[A.nDims - 1], a_T_stride[A.nDims - 2]);

    Tensor_view a_T = A;
    a_T.shape = a_T_shape.data();
    a_T.stride = a_T_stride.data();

    std::vector<int> b_T_shape(B.nDims);
    std::vector<int> b_T_stride(B.nDims);

    std::copy(B.shape, B.shape + B.nDims, b_T_shape.data());
    std::swap(b_T_shape[B.nDims - 1], b_T_shape[B.nDims - 2]);
    std::copy(B.stride, B.stride + B.nDims, b_T_stride.data());
    std::swap(b_T_stride[B.nDims - 1], b_T_stride[B.nDims - 2]);

    Tensor_view b_T = B;
    b_T.shape = b_T_shape.data();
    b_T.stride = b_T_stride.data();

    metadata_impl(A, B, C, this->strideA[0], this->strideB[0], this->strideC[0],
                  this->c_shape[0], this->A_colStride[0], this->B_rowStride[0],
                  this->shared_dim[0], this->C_nDims);
    metadata_impl(C, b_T, A, this->strideC[1], this->strideB[1],
                  this->strideA[1], this->c_shape[1], this->A_colStride[1],
                  this->B_rowStride[1], this->shared_dim[1], this->C_nDims);
    metadata_impl(a_T, C, B, this->strideA[2], this->strideC[2],
                  this->strideB[2], this->c_shape[2], this->A_colStride[2],
                  this->B_rowStride[2], this->shared_dim[2], this->C_nDims);

    // assign compile-time recursive function
    if (this->C_nDims <= Dispatchers::MAX_NDIMS) {
        this->forward_op =
            Dispatchers::get_batch_matmul<Scalar>()[this->C_nDims];
        this->backward_op =
            Dispatchers::get_batch_matmul_grad<Scalar>()[this->C_nDims];
    }
}

const char *Node_batch_matmul::node_type() const noexcept {
    return "Node_batch_matmul";
}

Node_batch_matmul::~Node_batch_matmul() {
    for (int i = 0; i < 3; i++) {
        delete[] strideA[i];
        delete[] strideB[i];
        delete[] strideC[i];
        delete[] c_shape[i];
    }
}

void Node_batch_matmul::eval() {
    if (!this->evaluated) {
        this->A->eval();
        this->B->eval();

        forward_op(this->A->value.data(), this->B->value.data(),
                   this->value.elements_.data(), strideA[0], strideB[0],
                   strideC[0], c_shape[0], A_colStride[0], B_rowStride[0],
                   shared_dim[0], C_nDims);
        this->evaluated = true;
    }
}

void Node_batch_matmul::getGrad() {
    backward_op(this->A->value.data(), this->A->gradient.elements_.data(),
                this->B->value.data(), this->B->gradient.elements_.data(),
                this->value.data(), this->gradient.data(), strideA + 1,
                strideB + 1, strideC + 1, c_shape + 1, A_colStride + 1,
                B_rowStride + 1, shared_dim + 1, C_nDims);

    if (this->A->hasInputs) {
        this->A->getGrad();
    }
    if (this->B->hasInputs) {
        this->B->getGrad();
    }
}

} // namespace kaad
