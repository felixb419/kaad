#pragma once

#include <algorithm>              // for fill, max, move
#include <cstddef>                // for size_t
#include <kaad/tensor/tensor.hpp> // for Tensor
#include <utility>                // for cmp_less_equal
#include <vector>                 // for vector

namespace kaad::detail {

/**
 * @brief Computes the resulting shape of matrix multiplication using
 * broadcasting.
 * Supports matmul-style broadcasting:
 *   (n?,k) x (k,m?) => (n?,m?)
 * @param shape1 Shape of first matrix.
 * @param rank1 Number of dimensions of first matrix.
 * @param shape2 Shape of second matrix.
 * @param rank2 Number of dimensions of second matrix.
 * @param newShape Output array to hold the result shape.
 * @param newLen Total number of dimensions in the result.
 * @return true if matmul broadcasting is possible, false otherwise.
 */
inline bool combine_matrix(const int *shape1, std::size_t rank1,
                           const int *shape2, std::size_t rank2, int *newShape,
                           std::size_t newLen) {
    if (shape1[rank1 - 1] != shape2[rank2 - 2]) {
        return false;
    }
    std::fill(newShape, newShape + newLen, 0);

    newShape[newLen - 1] = shape2[rank2 - 1];
    newShape[newLen - 2] = shape1[rank1 - 2];

    std::size_t ind = newLen - 3;
    for (int i = 3; std::cmp_less_equal(i, newLen); i++, ind--) {
        int ind1 = static_cast<int>(rank1) - i;
        int ind2 = static_cast<int>(rank2) - i;
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
 * @brief Computes stride and offset metadata for operations along a
 * specific tensor dimension.
 * @param input Input tensor.
 * @param output Output tensor (e.g., reduced along `dim`).
 * @param dim The dimension along which the operation is applied.
 * @param input_rank (out) Number of dimensions in input.
 * @param input_offset (out) Array storing the maximum valid offset per
 * dimension for @p input.
 * @param input_stride (out) Stride array for @p input.
 * @param output_stride (out) Stride array for @p output, adjusted to zero along
 * `dim`.
 */
inline void along_dim_metadata_impl(Tensor &input, Tensor &output, int dim,
                                    std::size_t &input_rank,
                                    std::vector<std::size_t> &input_offset,
                                    std::vector<int> &input_stride,
                                    std::vector<int> &output_stride) {
    input_rank = input.rank();
    input_stride =
        std::vector<int>(input.stride().begin(), input.stride().end());

    // make sure stride[i] is 1 instead of 0 if shape[i] is 1 for
    // traversing in flexible function
    for (std::size_t i = 0; i < input_rank; i++) {
        if (input_stride[i] == 0 && input.shape()[i] == 1) {
            input_stride[i] = 1;
        }
    }

    output_stride =
        std::vector<int>(output.stride().begin(), output.stride().end());

    // adjust output_stride if keep_rank was not set
    if (input.rank() > output.rank()) {

        output_stride.push_back(0);

        std::move(output_stride.begin() + dim, output_stride.end() - 1,
                  output_stride.begin() + dim + 1);
    }

    output_stride[dim] = 0;
    for (int i = 0; i < dim; i++) {
        output_stride[i] /= input.shape()[dim];
    }

    input_offset.resize(input_rank);
    for (std::size_t i = 0; i < input_rank; i++) {
        input_offset[i] =
            static_cast<std::size_t>(input.shape()[i]) * input_stride[i];
    }
}

} // namespace kaad::detail
