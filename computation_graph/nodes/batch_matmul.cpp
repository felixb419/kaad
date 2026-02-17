#include "batch_matmul.hpp"

#include "../../tensor/tensor_view.hpp" // for Tensor_view
#include "../common.hpp"                // for combine_matrix
#include "../dispatchers.hpp" // for get_batch_matmul, get_batch_matmul_grad

namespace kaad {

void metadata_impl(Tensor_view A, Tensor_view B, Tensor_view C, int *&strideA,
                   int *&strideB, int *&strideC, int *&c_shape_broadcast,
                   int &a_off, int &b_off, int &k, size_t &D) {
    a_off = A.stride[A.rank - 1];
    b_off = B.stride[B.rank - 2];
    k = A.shape[A.rank - 1];

    D = std::max(A.rank, B.rank);
    c_shape_broadcast = new int[D];

    detail::combine_matrix(A.shape, A.rank, B.shape, B.rank, c_shape_broadcast,
                           D);

    strideA = new int[D];
    strideB = new int[D];
    strideC = new int[D];

    int idx, idxA, idxB, idxC;
    for (int i = 1; i <= D; i++) {
        idx = D - i;
        idxA = A.rank - i;
        strideA[idx] = idxA >= 0 ? A.stride[idxA] : 0;
        idxB = B.rank - i;
        strideB[idx] = idxB >= 0 ? B.stride[idxB] : 0;
        idxC = C.rank - i;
        strideC[idx] = idxC >= 0 ? C.stride[idxC] : 0;
    }

    strideA[D - 1] = 0;
    strideB[D - 2] = 0;
}

void node_batch_matmul_metadata(Node_batch_matmul &node) {
    // compute metadata
    Tensor_view A = node.A->value.view();
    Tensor_view B = node.B->value.view();
    Tensor_view C = node.value.view();

    std::vector<int> a_T_shape(A.rank);
    std::vector<int> a_T_stride(A.rank);

    std::copy(A.shape, A.shape + A.rank, a_T_shape.data());
    std::swap(a_T_shape[A.rank - 1], a_T_shape[A.rank - 2]);
    std::copy(A.stride, A.stride + A.rank, a_T_stride.data());
    std::swap(a_T_stride[A.rank - 1], a_T_stride[A.rank - 2]);

    Tensor_view a_T = A;
    a_T.shape = a_T_shape.data();
    a_T.stride = a_T_stride.data();

    std::vector<int> b_T_shape(B.rank);
    std::vector<int> b_T_stride(B.rank);

    std::copy(B.shape, B.shape + B.rank, b_T_shape.data());
    std::swap(b_T_shape[B.rank - 1], b_T_shape[B.rank - 2]);
    std::copy(B.stride, B.stride + B.rank, b_T_stride.data());
    std::swap(b_T_stride[B.rank - 1], b_T_stride[B.rank - 2]);

    Tensor_view b_T = B;
    b_T.shape = b_T_shape.data();
    b_T.stride = b_T_stride.data();

    metadata_impl(A, B, C, node.strideA[0], node.strideB[0], node.strideC[0],
                  node.c_shape_broadcast[0], node.A_colStride[0],
                  node.B_rowStride[0], node.shared_dim[0], node.C_rank);
    metadata_impl(C, b_T, A, node.strideC[1], node.strideB[1], node.strideA[1],
                  node.c_shape_broadcast[1], node.A_colStride[1],
                  node.B_rowStride[1], node.shared_dim[1], node.C_rank);
    metadata_impl(a_T, C, B, node.strideA[2], node.strideC[2], node.strideB[2],
                  node.c_shape_broadcast[2], node.A_colStride[2],
                  node.B_rowStride[2], node.shared_dim[2], node.C_rank);

    // assign compile-time recursive function
    if (node.C_rank <= Dispatchers::MAX_NDIMS) {
        node.forward_op = Dispatchers::get_batch_matmul<Scalar>()[node.C_rank];
        node.backward_op =
            Dispatchers::get_batch_matmul_grad<Scalar>()[node.C_rank];
    }
}

const char *Node_batch_matmul::node_type() const noexcept {
    return "Node_batch_matmul";
}

Node_batch_matmul::Node_batch_matmul(INode *A_ptr, INode *B_ptr,
                                     std::span<const int> value_shape)
    : A(A_ptr), B(B_ptr), INode(value_shape, false) {

    node_batch_matmul_metadata(*this);
}

Node_batch_matmul::~Node_batch_matmul() noexcept {
    for (int i = 0; i < 3; i++) {
        delete[] strideA[i];
        delete[] strideB[i];
        delete[] strideC[i];
        delete[] c_shape_broadcast[i];
    }
}

void Node_batch_matmul::eval() {
    if (!this->evaluated) {
        this->A->eval();
        this->B->eval();

        forward_op(this->A->value.data(), this->B->value.data(),
                   this->value.elements_.data(), strideA[0], strideB[0],
                   strideC[0], c_shape_broadcast[0], A_colStride[0],
                   B_rowStride[0], shared_dim[0], C_rank);
        this->evaluated = true;
    }
}

void Node_batch_matmul::getGrad() {
    backward_op(this->A->value.data(), this->A->gradient.elements_.data(),
                this->B->value.data(), this->B->gradient.elements_.data(),
                this->value.data(), this->gradient.data(), strideA + 1,
                strideB + 1, strideC + 1, c_shape_broadcast + 1,
                A_colStride + 1, B_rowStride + 1, shared_dim + 1, C_rank);

    if (this->A->hasInputs) {
        this->A->getGrad();
    }
    if (this->B->hasInputs) {
        this->B->getGrad();
    }
}

} // namespace kaad
