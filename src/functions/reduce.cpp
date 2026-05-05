#include <kaad/functions/reduce.hpp>

#include <algorithm>                    // for copy, __copy_fn
#include <kaad/exceptions.hpp>          // for ArgumentError, to_string
#include <kaad/graph/inode.hpp>         // for INode
#include <kaad/scalar.hpp>              // for Scalar
#include <kaad/static_vector.hpp>       // for StaticVector
#include <kaad/tensor/tensor.hpp>       // for Tensor
#include <kaad/tensor/tensor_types.hpp> // for Shape, Strides
#include <kaad/tensor/tensor_view.hpp>  // for TensorViewConst
#include <string>                       // for allocator, char_traits, oper...

namespace kaad::functions::internal {

Shape make_res_shape_impl(std::array<INode *, 1> input, std::size_t dim,
                          bool keep_rank) {

    TensorViewConst inp = input[0]->value();

    if (std::cmp_greater_equal(dim, inp.rank())) {

        throw ArgumentError("dim has to be a valid index of A.shape, A.shape=" +
                            to_string(inp.shape) +
                            ", dim=" + std::to_string(dim));
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
        res[dim] = 1;

    } else {

        res.resize(res.size() - 1);

        auto cast_dim = static_cast<ptrdiff_t>(dim);

        std::copy(inp.shape.begin(), inp.shape.begin() + cast_dim, res.begin());
        std::copy(inp.shape.begin() + cast_dim + 1, inp.shape.end(),
                  res.begin() + dim);
    }

    return res;
}

void fwdparams_ctr_impl(const Scalar *&inp_begin, Scalar *&res_begin,
                        const Scalar *&res_end, Strides &eff_inp,
                        Strides &eff_res, Shape &inp_shape,
                        Scalar &relevant_dim_extent,
                        std::array<INode *, 1> input, INode *result,
                        std::size_t relevant_dim, bool keep_rank) {

    TensorViewConst inp = input[0]->value();
    TensorViewConst res = result->value();

    inp_begin = input[0]->value().data();

    res_begin = result->value_mut().data();
    res_end = result->value().data() + result->value().size();

    eff_inp = inp.strides;

    if (keep_rank) {

        eff_res = res.strides;
        eff_res[relevant_dim] = 0;

    } else {

        Shape res_padded = inp.shape;
        res_padded[relevant_dim] = 1;

        eff_res = Tensor::compute_strides(res_padded);
    }

    inp_shape = inp.shape;

    relevant_dim_extent = static_cast<Scalar>(input[0]->shape()[relevant_dim]);
}

} // namespace kaad::functions::internal
