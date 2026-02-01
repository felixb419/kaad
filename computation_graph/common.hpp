#pragma once

#include "../tensor/tensor_view.hpp" // for Tensor_view
#include <algorithm>                 // for fill, max
#include <stddef.h>                  // for size_t

namespace kaad::detail {

/**
 * @brief Computes the resulting shape of matrix multiplication using
 * broadcasting.
 * Supports matmul-style broadcasting:
 *   (n?,k) x (k,m?) => (n?,m?)
 * @param shape1 Shape of first matrix.
 * @param nDims1 Number of dimensions of first matrix.
 * @param shape2 Shape of second matrix.
 * @param nDims2 Number of dimensions of second matrix.
 * @param newShape Output array to hold the result shape.
 * @param newLen Total number of dimensions in the result.
 * @return true if matmul broadcasting is possible, false otherwise.
 */
static inline bool combine_matrix(const int *shape1, size_t nDims1,
                                  const int *shape2, size_t nDims2,
                                  int *newShape, size_t newLen) {
    if (shape1[nDims1 - 1] != shape2[nDims2 - 2]) {
        return false;
    }
    std::fill(newShape, newShape + newLen, 0);

    newShape[newLen - 1] = shape2[nDims2 - 1];
    newShape[newLen - 2] = shape1[nDims1 - 2];

    int ind = newLen - 3;
    for (int i = 3; i <= newLen; i++, ind--) {
        int ind1 = nDims1 - i;
        int ind2 = nDims2 - i;
        if (ind1 >= 0 && ind2 >= 0) {
            if (shape1[ind1] != shape2[ind2] && shape1[ind1] != 1 &&
                shape2[ind2] != 1) {
                return false;
            }
            newShape[ind] = std::max(shape1[ind1], shape2[ind2]);
        } else {
            newShape[ind] = ind1 >= 0 ? shape1[ind1] : shape2[ind2];
        }
    }
    return true;
}

/**
 * @brief Computes stride and offset metadata for operations along a specific
 * tensor dimension.
 * @param A          Input tensor.
 * @param C          Output tensor (e.g., reduced along `dim`).
 * @param dim        The dimension along which the operation is applied.
 * @param D          (out) Number of dimensions.
 * @param A_offset   (out) Array storing the maximum valid offset per dimension
 * for A.
 * @param strideA    (out) Stride array for A.
 * @param strideC    (out) Stride array for C, adjusted to zero along `dim`.
 */
void along_dim_metadata_impl(Tensor_view &A, Tensor_view &C, int dim, size_t &D,
                             std::vector<size_t> &A_offset,
                             std::vector<int> &strideA,
                             std::vector<int> &strideC) {
    D = A.nDims;
    strideA.resize(D);
    strideC.resize(D);

    std::copy(A.stride, A.stride + A.nDims, strideA.data());
    std::copy(A.stride, A.stride + A.nDims, strideC.data());
    // make sure stride[i] is 1 instead of 0 if shape[i] is 1 for
    // traversing in flexible function
    for (int i = 0; i < D; i++) {
        if (strideA[i] == 0 && A.shape[i] == 1) {
            strideA[i] = 1;
        }
    }

    strideC[dim] = 0;
    for (int i = 0; i < dim; i++) {
        strideC[i] /= A.shape[dim];
    }

    A_offset.resize(D);
    for (int i = 0; i < D; i++) {
        A_offset[i] = A.shape[i] * strideA[i];
    }
}

} // namespace kaad::detail
