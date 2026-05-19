#include "reduce.hpp"

#include "kaad/graph/internal/inode.hpp"
#include "kaad/tensor/internal/tensor.hpp"
#include "kaad/tensor/internal/tensor_types.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <kaad/exceptions.hpp>
#include <kaad/scalar.hpp>
#include <string>
#include <utility>

namespace kaad::operations::internal {

Shape make_res_shape_impl(std::array<INode *, 1> input, std::size_t axis,
                          bool keep_rank) {

    const Tensor &inp = input[0]->value;

    if (std::cmp_greater_equal(axis, inp.rank())) {

        throw ArgumentError(
            "axis has to be a valid index of A.shape, A.shape=" +
            to_string(inp.shape) + ", axis=" + std::to_string(axis));
    }

    if (inp.rank() == 1) {

        throw ArgumentError(
            "input.rank() has to be grater than 1, input.rank()=" +
            std::to_string(inp.rank()));
    }

    std::size_t new_len = inp.rank();
    Shape res(new_len);
    if (keep_rank) {

        std::ranges::copy(inp.shape, res.begin());
        res[axis] = 1;

    } else {

        res.resize(res.size() - 1);

        auto cast_axis = static_cast<ptrdiff_t>(axis);

        std::copy(inp.shape.begin(), inp.shape.begin() + cast_axis,
                  res.begin());
        std::copy(inp.shape.begin() + cast_axis + 1, inp.shape.end(),
                  res.begin() + axis);
    }

    return res;
}

void fwdparams_ctr_impl(const Scalar *&inp_begin, Scalar *&res_begin,
                        const Scalar *&res_end, Strides &eff_inp,
                        Strides &eff_res, Shape &inp_shape,
                        Scalar &reduction_extent, std::array<INode *, 1> input,
                        INode *result, std::size_t reduction_axis,
                        bool keep_rank) {

    const Tensor &inp = input[0]->value;
    const Tensor &res = result->value;

    inp_begin = inp.data;

    res_begin = res.data;
    res_end = res.data + res.size;

    eff_inp = inp.strides;

    if (keep_rank) {

        eff_res = res.strides;
        eff_res[reduction_axis] = 0;

    } else {

        Shape res_padded = inp.shape;
        res_padded[reduction_axis] = 1;

        eff_res = Tensor::compute_strides(res_padded);
    }

    inp_shape = inp.shape;

    reduction_extent = static_cast<Scalar>(input[0]->shape()[reduction_axis]);
}

} // namespace kaad::operations::internal
