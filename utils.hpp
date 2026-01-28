#pragma once

#include "tensor/tensor.hpp"
#include <iostream> // for std::ostream, std::cout, std::endl

namespace kaad {

/**
 * @brief Computes the resulting shape from broadcasting two tensors.
 *
 * The broadcasting rules are:
 *   - dimensions must match
 *   - or one of them must be 1
 *
 * @param shape1 Shape of first tensor.
 * @param nDims1 Number of dimensions of first tensor.
 * @param shape2 Shape of second tensor.
 * @param nDims2 Number of dimensions of second tensor.
 * @param newShape Output array to hold the result shape.
 * @param newLen Total number of dimensions in the result.
 * @return true if broadcasting is possible, false otherwise.
 */
static inline bool combine_flexible(const int *shape1, size_t nDims1,
                                    const int *shape2, size_t nDims2,
                                    int *newShape, size_t newLen) {
    int ind = newLen - 1;
    for (int i = 1; i <= newLen; i++, ind--) {
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
 * @brief Computes the resulting shape of matrix multiplication using
 * broadcasting.
 *
 * Supports matmul-style broadcasting:
 *   (n?,k) x (k,m?) => (n?,m?)
 *
 * @param shape1 Shape of first matrix.
 * @param nDims1 Number of dimensions of first matrix.
 * @param shape2 Shape of second matrix.
 * @param nDims2 Number of dimensions of second matrix.
 * @param newShape Output array to hold the result shape.
 * @param newLen Total number of dimensions in the result.
 * @return true if matmul broadcasting is possible, false otherwise.
 */
bool combine_matrix(const int *shape1, size_t nDims1, const int *shape2,
                    size_t nDims2, int *newShape, size_t newLen) {
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

} // namespace kaad
