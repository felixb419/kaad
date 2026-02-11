#include "matmul.hpp"

#include <algorithm> // for std::reverse_copy

namespace kaad {

/**
 * @brief Prepares stride and shape parameters for a 2D matrix multiplication.
 * @param A         View of the left-hand side matrix.
 * @param B         View of the right-hand side matrix.
 * @param C         View of the output matrix.
 * @param a_dim     Output variable for the number of rows in A.
 * @param b_dim     Output variable for the number of columns in B.
 * @param k         Output variable for the shared inner dimension.
 * @param strideA   Output stride array for A.
 * @param strideB   Output stride array for B.
 * @param strideC   Output stride array for C, modified for efficient traversal.
 */
void metadata_impl(const Tensor_view A, const Tensor_view B,
                   const Tensor_view C, int &a_dim, int &b_dim, int &k,
                   int *strideA, int *strideB, int *strideC) {
    a_dim = A.shape[0];
    b_dim = B.shape[1];
    k = A.shape[1];

    std::copy(A.stride, A.stride + 2, strideA);
    std::copy(B.stride, B.stride + 2, strideB);
    std::copy(C.stride, C.stride + 2, strideC);

    int idx, idxA, idxB, idxC;
    int offsetC = 0, prevC;
    for (int i = 1; i <= 2; i++) {
        idx = 2 - i;

        idxC = C.rank - i;
        prevC = offsetC;
        offsetC += ((idxC >= 0 ? C.shape[idxC] : i) - 1) * strideC[idx];
        strideC[idx] -= prevC + strideC[idx + 1];
    }
}

void Node_matmul::metadata() {
    // compute metadata
    Tensor_view A = this->A->value.view();
    Tensor_view B = this->B->value.view();
    Tensor_view C = this->value.view();

    int A_T_shape[2];
    int A_T_stride[2];
    std::reverse_copy(A.shape, A.shape + A.rank, A_T_shape);
    std::reverse_copy(A.stride, A.stride + A.rank, A_T_stride);

    Tensor_view A_T = A;
    A_T.shape = A_T_shape;
    A_T.stride = A_T_stride;

    int B_T_shape[2];
    int B_T_stride[2];
    std::reverse_copy(B.shape, B.shape + B.rank, B_T_shape);
    std::reverse_copy(B.stride, B.stride + B.rank, B_T_stride);

    Tensor_view B_T = B;
    B_T.shape = B_T_shape;
    B_T.stride = B_T_stride;

    metadata_impl(A, B, C, this->a_rows[0], this->b_cols[0],
                  this->shared_dim[0], this->strideA, this->strideB,
                  this->strideC);
    metadata_impl(C, B_T, A, this->a_rows[1], this->b_cols[1],
                  this->shared_dim[1], this->strideC + 2, this->strideB + 2,
                  this->strideA + 2);
    metadata_impl(A_T, C, B, this->a_rows[2], this->b_cols[2],
                  this->shared_dim[2], this->strideA + 4, this->strideC + 4,
                  this->strideB + 4);
}

const char *Node_matmul::node_type() const noexcept { return "Node_matmul"; }

void Node_matmul::eval() {
    if (!this->evaluated) {
        this->A->eval();
        this->B->eval();

        forward_op(this->A->value.data(), this->B->value.data(),
                   this->value.elements_.data(), a_rows[0], b_cols[0],
                   shared_dim[0], strideA, strideB, strideC);
        this->evaluated = true;
    }
}

void Node_matmul::getGrad() {
    backward_op(this->A->value.data(), this->A->gradient.elements_.data(),
                this->B->value.data(), this->B->gradient.elements_.data(),
                this->value.data(), this->gradient.data(), a_rows + 1,
                b_cols + 1, shared_dim + 1, strideA + 2, strideB + 2,
                strideC + 2);

    if (this->A->hasInputs) {
        this->A->getGrad();
    }
    if (this->B->hasInputs) {
        this->B->getGrad();
    }
}

} // namespace kaad
