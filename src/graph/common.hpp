#pragma once

#include <algorithm>              // for fill, max, move
#include <cstddef>                // for size_t
#include <kaad/tensor/tensor.hpp> // for Tensor
#include <utility>                // for cmp_less_equal
#include <vector>                 // for vector

namespace kaad::detail {

/**
 * @brief Computes strides and offset metadata for operations along a
 * specific tensor dimension.
 * @param input Input tensor.
 * @param output Output tensor (e.g., reduced along `dim`).
 * @param dim The dimension along which the operation is applied.
 * @param input_rank (out) Number of dimensions in input.
 * @param input_offset (out) Array storing the maximum valid offset per
 * dimension for @p input.
 * @param input_strides (out) Stride array for @p input.
 * @param output_strides (out) Stride array for @p output, adjusted to zero
 * along `dim`.
 */
inline void along_dim_metadata_impl(Tensor &input, Tensor &output, int dim,
                                    std::size_t &input_rank,
                                    StaticVector<std::size_t> &input_offset,
                                    Strides &input_strides,
                                    Strides &output_strides) {
    input_rank = input.rank();
    input_strides = Strides(input.strides());

    // make sure strides[i] is 1 instead of 0 if shape[i] is 1 for
    // traversing in flexible function
    for (std::size_t i = 0; i < input_rank; i++) {
        if (input_strides[i] == 0 && input.shape()[i] == 1) {
            input_strides[i] = 1;
        }
    }

    output_strides = Strides(output.strides());

    // adjust output_strides if keep_rank was not set
    if (input.rank() > output.rank()) {

        output_strides.push_back(0);

        std::move(output_strides.begin() + dim, output_strides.end() - 1,
                  output_strides.begin() + dim + 1);
    }

    output_strides[dim] = 0;
    for (int i = 0; i < dim; i++) {
        output_strides[i] /= input.shape()[dim];
    }

    input_offset.resize(input_rank);
    for (std::size_t i = 0; i < input_rank; i++) {
        input_offset[i] =
            static_cast<std::size_t>(input.shape()[i]) * input_strides[i];
    }
}

} // namespace kaad::detail
