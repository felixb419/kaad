#include <kaad/functions/batch_matmul.hpp>

#include <algorithm>                    // for __copy_fn, copy, max
#include <kaad/tensor/tensor_types.hpp> // for ShapeView, Shape
#include <kaad/tensor/tensor_view.hpp>  // for TensorViewConst
#include <utility>                      // for cmp_less_equal

namespace kaad::functions {

bool BatchMatmul::broadcast(ShapeView lhs, ShapeView rhs, Shape &new_shape) {
    size_t lhs_rank = lhs.size();
    size_t rhs_rank = rhs.size();
    size_t new_rank = std::max(lhs_rank, rhs_rank);

    if (lhs[lhs_rank - 1] != rhs[rhs_rank - 2]) {
        return false;
    }

    new_shape.resize(new_rank);

    new_shape[new_rank - 1] = rhs[rhs_rank - 1];
    new_shape[new_rank - 2] = lhs[lhs_rank - 2];

    std::size_t idx_new = new_rank - 3;
    for (int i = 3; std::cmp_less_equal(i, new_rank); i++, idx_new--) {
        int idx_lhs = static_cast<int>(lhs_rank) - i;
        int idx_rhs = static_cast<int>(rhs_rank) - i;
        if (idx_lhs >= 0 && idx_rhs >= 0) {
            if (lhs[idx_lhs] != rhs[idx_rhs] && lhs[idx_lhs] != 1 &&
                rhs[idx_rhs] != 1) {
                return false;
            }
            new_shape[idx_new] = std::max(lhs[idx_lhs], rhs[idx_rhs]);
        } else {
            new_shape[idx_new] = idx_lhs >= 0 ? lhs[idx_lhs] : rhs[idx_rhs];
        }
    }
    return true;
}

BatchMatmul::Metadata::Metadata(TensorViewConst lhs, TensorViewConst rhs,
                                TensorViewConst res) {

    this->lhs_col_step = lhs.stride[lhs.rank() - 1];
    this->rhs_row_step = rhs.stride[rhs.rank() - 2];
    this->shared_dim = lhs.shape[lhs.rank() - 1];

    broadcast(lhs.shape, rhs.shape, this->res_broadcast);

    std::size_t res_rank = res_broadcast.size();

    // copying strides into effective strides starting from the back
    this->eff_lhs.resize(res_rank);
    int lhs_rank_diff = static_cast<int>(res_rank - lhs.rank());
    std::ranges::copy(lhs.stride, eff_lhs.begin() + lhs_rank_diff);

    this->eff_rhs.resize(res_rank);
    int rhs_rank_diff = static_cast<int>(res_rank - rhs.rank());
    std::ranges::copy(rhs.stride, eff_rhs.begin() + rhs_rank_diff);

    this->eff_res.resize(res_rank);
    int res_rank_diff = static_cast<int>(res_rank - res.rank());
    std::ranges::copy(res.stride, eff_res.begin() + res_rank_diff);

    eff_lhs[res_rank - 1] = 0;
    eff_rhs[res_rank - 2] = 0;
}

} // namespace kaad::functions
