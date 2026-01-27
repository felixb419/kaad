#pragma once

#include "tensor/tensor.hpp"
#include <iostream> // for std::ostream, std::cout, std::endl

namespace kaad {

/**
 * @brief Recursively prints a tensor in nested bracket format.
 *
 * @tparam T Data type of the tensor.
 * @param os Output stream to write to.
 * @param cords Current coordinates (index per dimension).
 * @param tensor Tensor to print.
 * @param ind Current dimension index being printed.
 */
template <typename T>
void print_flat_impl(std::ostream &os, int *cords, const Tensor<T> &tensor,
                     int ind) {
    if (ind == tensor.nDims) {
        os << tensor.val[getIndex(cords, tensor.shape, tensor.stride,
                                  tensor.nDims)];
    } else {
        os << "[";
        int lim = tensor.shape[ind];
        // iterate for size of current dimension
        for (int i = 0; i < lim - 1; i++) {
            // print next dimension
            print_flat_impl(os, cords, tensor, ind + 1);
            os << ",";
            cords[ind]++;
        }
        // last pass without trailing comma
        print_flat_impl(os, cords, tensor, ind + 1);
        cords[ind]++;

        os << "]";
    }
}

/**
 * @brief Prints the contents of a tensor in flat, human-readable format.
 *
 * Example output:
 * @code
 * [[1,2],[3,4]]
 * @endcode
 *
 * @tparam T Data type of the tensor.
 * @param tensor Tensor to print.
 * @param stream Output stream (defaults to std::cout).
 */
template <typename T>
void print_flat(const Tensor<T> &tensor, std::ostream &stream = std::cout) {
    int *cords = new int[tensor.nDims];
    std::fill(cords, cords + tensor.nDims, 0);

    print_flat_impl(stream, cords, tensor, 0);

    stream << std::endl;
    delete[] cords;
}

/**
 * @brief Prints a flat array to the output stream.
 *
 * @tparam T Data type of the array.
 * @param begin Pointer to the beginning of the array.
 * @param end Pointer to one past the end of the array.
 * @param os Output stream (defaults to std::cout).
 */
template <typename T>
inline void print_arr(const T *begin, const T *end,
                      std::ostream &os = std::cout) {
    os << "[";
    for (const T *p = begin; p != end; p++) {
        if (p != begin) {
            os << ",";
        }
        os << *p;
    }
    os << "]";
}

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
