#include <kaad/functions/matmul.hpp>

#include <algorithm>                    // for __copy_fn, copy, max
#include <kaad/exceptions.hpp>          // for BroadcastError, to_string
#include <kaad/tensor/tensor_types.hpp> // for Shape, ShapeView, extent
#include <kaad/tensor/tensor_view.hpp>  // for TensorViewConst
#include <string>                       // for allocator, char_traits, oper...

namespace kaad::functions {

Shape Matmul::broadcast(ShapeView lhs, ShapeView rhs) {
    size_t lhs_rank = lhs.size();
    size_t rhs_rank = rhs.size();
    size_t new_rank = std::max(lhs_rank, rhs_rank);

    if (lhs[lhs_rank - 1] != rhs[rhs_rank - 2]) {

        throw BroadcastError("incompatible tensor shapes for matrix "
                             "multiplication, lhs.shape()" +
                             to_string(lhs) + ", rhs.shape()" + to_string(rhs));
    }

    Shape res(new_rank);

    res[new_rank - 1] = rhs[rhs_rank - 1];
    res[new_rank - 2] = lhs[lhs_rank - 2];

    // check batch dimensions
    if (new_rank > 2) {

        auto broadcast_compatible = [](auto dim1, auto dim2) {
            return dim1 == dim2 || dim1 == 1 || dim2 == 1;
        };

        // +2 in offset because last two dimensions are already checked
        for (std::size_t offset = 1 + 2; offset <= new_rank; offset++) {

            std::size_t lhs_idx = lhs.size() - offset;
            std::size_t rhs_idx = rhs.size() - offset;
            std::size_t res_idx = rhs.size() - offset;

            if (offset > lhs.size()) {
                res[res_idx] = rhs[rhs_idx];
                continue;
            }

            if (offset > rhs.size()) {
                res[res_idx] = lhs[lhs_idx];
                continue;
            }

            if (!broadcast_compatible(lhs[lhs_idx], rhs[rhs_idx])) {

                throw BroadcastError(
                    "incompatible tensor shapes for broadcasting, lhs.shape()" +
                    to_string(lhs) + ", rhs.shape()" + to_string(rhs));
            }

            res[res_idx] = std::max(lhs[lhs_idx], rhs[rhs_idx]);
        }
    }

    return res;
}

Matmul::Metadata::Metadata(TensorViewConst lhs, TensorViewConst rhs,
                           TensorViewConst res) {

    this->lhs_col_step = lhs.strides[lhs.rank() - 1];
    this->rhs_row_step = rhs.strides[rhs.rank() - 2];
    this->shared_dim = lhs.shape[lhs.rank() - 1];

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

    eff_lhs[res_rank - 1] = 0;
    eff_rhs[res_rank - 2] = 0;
}

} // namespace kaad::functions
