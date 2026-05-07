#include <kaad/operations/flexible.hpp>

#include <algorithm>                    // for max
#include <kaad/exceptions.hpp>          // for BroadcastError, to_string
#include <kaad/graph/inode.hpp>         // for INode
#include <kaad/tensor/tensor_types.hpp> // for Strides, Shape, ShapeView
#include <kaad/tensor/tensor_view.hpp>  // for TensorViewConst, TensorViewMut
#include <string>                       // for allocator, char_traits, oper...

namespace kaad::operations {

Shape BroadcastPolicy::make_res_shape(std::array<INode *, 2> inputs) {

    ShapeView lhs = inputs[0]->shape();
    ShapeView rhs = inputs[1]->shape();

    std::size_t new_rank = std::max(lhs.size(), rhs.size());

    Shape res(new_rank);

    auto broadcast_compatible = [](auto extent1, auto extent2) {
        return extent1 == extent2 || extent1 == 1 || extent2 == 1;
    };

    for (std::size_t offset = 1; offset <= new_rank; offset++) {

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

    return res;
}

void BroadcastPolicy::init_strides(std::array<INode *, 2> inputs, INode *result,
                                   Strides &eff_lhs, Strides &eff_rhs,
                                   Strides &eff_res) {

    TensorViewConst lhs = inputs[0]->value();
    TensorViewConst rhs = inputs[1]->value();
    TensorViewMut res = result->value_mut();

    std::size_t idx_lhs;
    std::size_t idx_rhs;
    std::size_t idx_res;
    for (std::size_t i = 1; i <= res.rank(); i++) {

        idx_res = res.rank() - i;
        eff_res[idx_res] = i <= res.rank() ? res.strides[idx_res] : 0;

        idx_lhs = lhs.rank() - i;
        eff_lhs[idx_res] = i <= lhs.rank() ? lhs.strides[idx_lhs] : 0;

        idx_rhs = rhs.rank() - i;
        eff_rhs[idx_res] = i <= rhs.rank() ? rhs.strides[idx_rhs] : 0;

        // make sure eff_res[idx] is 1 instead of 0 if
        // res.shape[idx] is 1 for traversing in flexible function
        if (eff_res[idx_res] == 0 && res.shape[idx_res] == 1) {
            eff_res[idx_res] = 1;
        }
    }
}

} // namespace kaad::operations
