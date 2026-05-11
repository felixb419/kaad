#include "matmul.hpp"

#include "strided.hpp"                   // for BroadcastPolicy
#include <algorithm>                     // for __copy_fn, copy, max
#include <array>                         // for array
#include <cstddef>                       // for size_t
#include <kaad/exceptions.hpp>           // for BroadcastError, to_string
#include <kaad/graph/internal/inode.hpp> // for INode
#include <kaad/tensor/internal/tensor_types.hpp> // for Shape, ShapeView, Strides
#include <kaad/tensor/tensor_view.hpp> // for TensorViewConst, TensorViewMut
#include <span>                        // for span

namespace kaad::operations {

Shape broadcast(ShapeView lhs, ShapeView rhs) {

    size_t lhs_rank = lhs.size();
    size_t rhs_rank = rhs.size();
    size_t res_rank = std::max(lhs_rank, rhs_rank);

    if (lhs[lhs.size() - 1] != rhs[rhs.size() - 2]) {

        throw BroadcastError("incompatible tensor shapes for matrix "
                             "multiplication, lhs.shape()" +
                             to_string(lhs) + ", rhs.shape()" + to_string(rhs));
    }

    if (res_rank > 2) {

        ShapeView batch_axes_lhs(lhs.data(), lhs.size() - 2);
        ShapeView batch_axes_rhs(rhs.data(), rhs.size() - 2);

        Shape batch_axes_broadcast =
            BroadcastPolicy::make_res_shape(batch_axes_lhs, batch_axes_rhs);

        batch_axes_broadcast.push_back(lhs[lhs.size() - 2]);
        batch_axes_broadcast.push_back(rhs[rhs.size() - 1]);

        return batch_axes_broadcast;
    }

    return {lhs[lhs.size() - 2], rhs[rhs.size() - 1]};
}

Shape Matmul::make_res_shape(std::array<INode *, 2> inputs) {
    return broadcast(inputs[0]->shape(), inputs[1]->shape());
}

Matmul::ForwardParams::ForwardParams(TensorViewConst lhs, TensorViewConst rhs,
                                     TensorViewMut res) {

    this->lhs_col_step = lhs.strides[lhs.rank() - 1];
    this->rhs_row_step = rhs.strides[rhs.rank() - 2];
    this->extent_shared = lhs.shape[lhs.rank() - 1];

    this->res_broadcast = broadcast(lhs.shape, rhs.shape);

    std::size_t res_rank = res_broadcast.size();

    // copying strides into effective strides starting from the back
    this->eff_lhs.resize(res_rank);
    int lhs_rank_diff = static_cast<int>(res_rank - lhs.rank());
    std::ranges::copy(lhs.strides, eff_lhs.begin() + lhs_rank_diff);

    this->eff_rhs.resize(res_rank);
    int rhs_rank_diff = static_cast<int>(res_rank - rhs.rank());
    std::ranges::copy(rhs.strides, eff_rhs.begin() + rhs_rank_diff);

    this->eff_res.resize(res_rank);
    int res_rank_diff = static_cast<int>(res_rank - res.rank());
    std::ranges::copy(res.strides, eff_res.begin() + res_rank_diff);

    eff_lhs.from_back(0) = 0;
    eff_rhs.from_back(1) = 0;

    this->lhs_begin = lhs.elements.data();
    this->rhs_begin = rhs.elements.data();
    this->res_begin = res.elements.data();
}

Matmul::ForwardParams::ForwardParams(std::array<INode *, 2> inputs,
                                     INode *result)
    : ForwardParams(inputs[0]->value(), inputs[1]->value(),
                    result->value_mut()) {}

Matmul::BackwardParams::BackwardParams(std::array<INode *, 2> inputs,
                                       INode *result) {

    TensorViewConst lhs_view = inputs[0]->value();
    TensorViewMut d_lhs_view = inputs[0]->gradient_mut();

    TensorViewConst rhs_view = inputs[1]->value();
    TensorViewMut d_rhs_view = inputs[1]->gradient_mut();

    TensorViewConst d_res_view = result->gradient();

    Shape lhs_sh_buff;
    Strides lhs_st_buff;
    TensorViewConst lhs_transposed =
        lhs_view.transpose_2d(lhs_sh_buff, lhs_st_buff);

    Shape rhs_sh_buff;
    Strides rhs_st_buff;
    TensorViewConst rhs_transposed =
        rhs_view.transpose_2d(rhs_sh_buff, rhs_st_buff);

    // d_res * rhs^T = d_lhs
    this->wrt_lhs = ForwardParams(d_res_view, rhs_transposed, d_lhs_view);

    // lhs^T * d_res = d_rhs
    this->wrt_rhs = ForwardParams(lhs_transposed, d_res_view, d_rhs_view);
}

} // namespace kaad::operations
