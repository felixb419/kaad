#include "slice.hpp"

#include <kaad/exceptions.hpp>                   // for ArgumentError, to_string
#include <kaad/graph/internal/inode.hpp>         // for INode
#include <kaad/static_vector.hpp>                // for StaticVector
#include <kaad/tensor/internal/tensor_types.hpp> // for ShapeView, Shape
#include <kaad/tensor/tensor_view.hpp>           // for TensorViewConst
#include <string> // for allocator, char_traits, oper...

namespace kaad::operations {

Shape Slice::make_res_shape(std::array<INode *, 1> input, ShapeView shape,
                            StaticVector<std::size_t> start) {

    TensorViewConst inp = input[0]->value();

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
                                    [[maybe_unused]] ShapeView shape,
                                    std::span<const std::size_t> start)
    : inp_begin(input[0]->value().data()),
      res_begin(result->value_mut().data()), eff_inp(input[0]->value().strides),
      eff_res(result->value().strides), res_shape(result->shape()) {

    this->inp_start_offset = start;
    for (std::size_t i = 0; i < start.size(); i++) {
        this->inp_start_offset[i] *= this->eff_inp[i];
    }
}

} // namespace kaad::operations
