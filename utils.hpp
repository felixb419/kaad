#pragma once

#include "tensor/tensor.hpp"
#include <iostream> // for std::ostream, std::cout, std::endl

namespace kaad {

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
