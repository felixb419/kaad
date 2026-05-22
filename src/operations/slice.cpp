#include "slice.hpp"
#include "kaad/graph/internal/inode.hpp"
#include "kaad/tensor/internal/tensor.hpp"
#include "kaad/tensor/internal/tensor_types.hpp"

#include <array>
#include <cstddef>
#include <kaad/exceptions.hpp>
#include <kaad/static_vector.hpp>
#include <string>

namespace kaad::operations {

Shape Slice::make_res_shape(std::array<INode *, 1> input,
                            const Metadata &mdata) {

    const Tensor &inp = input[0]->value;
    ShapeView shape = mdata.res_shape;
    Shape start = mdata.res_start;

    if (shape.size() != inp.rank()) {

        throw ArgumentError(
            "length of shape must be equal to input.rank(), shape=" +
            to_string(shape) + ", input.shape()=" + to_string(inp.shape));
    }

    if (start.size() > inp.rank()) {

        throw ArgumentError(
            "length of start must not be bigger than input.rank(), start=" +
            to_string(start) + ", input.shape" + to_string(inp.shape));
    }

    // pad start with 0s
    if (start.size() < inp.rank()) {
        start.resize(inp.rank());
    }

    // make sure slice is not too large
    for (std::size_t i = 0; i < inp.rank(); i++) {

        if (shape[i] + start[i] > inp.shape[i]) {

            throw ArgumentError("axis " + std::to_string(i) +
                                " of slice is too large with an extent of " +
                                std::to_string(shape[i]) + " starting at " +
                                std::to_string(start[i]));
        }
    }

    return {shape};
}

Slice::ForwardParams::ForwardParams(std::array<INode *, 1> input, INode *result,
                                    const Metadata &mdata)
    : inp_begin(input[0]->value.data), res_begin(result->value.data),
      eff_inp(input[0]->value.strides), eff_res(result->value.strides),
      res_shape(result->shape()) {

    this->inp_start_offset = mdata.res_start;
    for (std::size_t i = 0; i < mdata.res_start.size(); i++) {
        this->inp_start_offset[i] *= this->eff_inp[i];
    }
}

} // namespace kaad::operations
