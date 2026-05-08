#include "transpose.hpp"

#include <algorithm>                    // for __all_of_fn, __sort_fn, adja...
#include <kaad/exceptions.hpp>          // for ArgumentError, ShapeError
#include <kaad/graph/inode.hpp>         // for INode
#include <kaad/scalar.hpp>              // for Scalar
#include <kaad/static_vector.hpp>       // for StaticVector
#include <kaad/tensor/tensor_types.hpp> // for Shape, Strides
#include <kaad/tensor/tensor_view.hpp>  // for TensorViewConst
#include <ranges>                       // for __adjacent_find_fn
#include <string>                       // for allocator, char_traits, oper...

namespace kaad::operations {

bool contains_duplicates(StaticVector<std::size_t> vals) {
    std::ranges::sort(vals);
    return std::ranges::adjacent_find(vals) != vals.end();
}

bool all_below(std::span<const std::size_t> vals,
               std::size_t excl_upper_bound) {

    return std::ranges::all_of(vals, [excl_upper_bound](std::size_t val) {
        return val < excl_upper_bound;
    });
}

std::pair<Shape, Strides>
Transpose::make_res_shape(std::array<INode *, 1> input,
                          std::span<const std::size_t> perm) {

    TensorViewConst inp = input[0]->value();

    if (inp.rank() < 2) {
        throw ShapeError("input.rank() hast to be > 1, input.rank()=" +
                         std::to_string(inp.rank()));
    }

    if (perm.size() != inp.rank() || contains_duplicates(perm) ||
        !all_below(perm, inp.rank())) {

        throw ArgumentError(
            "perm is not a valid permutation of input.shape(), perm=" +
            to_string(perm) + ", input.shape()=" + to_string(inp.shape));
    }

    Shape res_shape(inp.rank());
    Strides res_strides(inp.rank());
    for (std::size_t i = 0; i < inp.rank(); i++) {

        res_shape[i] = inp.shape[perm[i]];
        res_strides[i] = inp.strides[perm[i]];
    }

    return {res_shape, res_strides};
}

Transpose::ForwardParams::ForwardParams(
    std::array<INode *, 1> input, INode *result,
    [[maybe_unused]] std::span<const std::size_t> perm)
    : inp_begin(input[0]->value().data()),
      inp_end(input[0]->value().data() + input[0]->value().size()),
      res_begin(result->value_mut().data()) {}

void Transpose::forward(const ForwardParams &params) {
    std::copy(params.inp_begin, params.inp_end, params.res_begin);
}

void Transpose::backward(const BackwardParams &params) {

    Scalar *d_inp = params.d_inp_begin;
    const Scalar *d_res = params.d_res_begin;

    for (; d_inp < params.d_inp_end; d_inp++, d_res++) {

        *d_inp += *d_res;
    }
}

Transpose::Dispatch
Transpose::dispatch([[maybe_unused]] std::array<INode *, 1> input,
                    [[maybe_unused]] INode *result,
                    [[maybe_unused]] std::span<const std::size_t> perm) {

    return {.forward = forward, .backward = backward};
}

} // namespace kaad::operations
