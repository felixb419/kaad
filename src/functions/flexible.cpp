#include <kaad/functions/flexible.hpp>

#include "kaad/exceptions.hpp"          // for BroadcastError, to_string
#include "kaad/static_vector.hpp"       // for StaticVector
#include "kaad/tensor/tensor_types.hpp" // for Shape, ShapeView
#include "kaad/tensor/tensor_view.hpp"  // for TensorView
#include <algorithm>                    // for max
#include <kaad/tensor/tensor.hpp>       // for TensorViewConst
#include <string>                       // for allocator, char_traits, oper...

namespace kaad::functions {

Shape Flexible::broadcast(ShapeView lhs, ShapeView rhs) {

    std::size_t new_rank = std::max(lhs.size(), rhs.size());

    Shape res(new_rank);

    auto broadcast_compatible = [](auto dim1, auto dim2) {
        return dim1 == dim2 || dim1 == 1 || dim2 == 1;
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

Flexible::Metadata::Metadata(TensorViewConst lhs, TensorViewConst rhs,
                             TensorViewConst res) {

    this->eff_lhs.resize(res.rank());
    this->eff_rhs.resize(res.rank());
    this->eff_res.resize(res.rank());

    std::size_t idx_lhs;
    std::size_t idx_rhs;
    std::size_t idx_res;
    for (std::size_t i = 1; i <= res.rank(); i++) {

        idx_res = res.rank() - i;
        this->eff_res[idx_res] = i <= res.rank() ? res.strides[idx_res] : 0;

        idx_lhs = lhs.rank() - i;
        this->eff_lhs[idx_res] = i <= lhs.rank() ? lhs.strides[idx_lhs] : 0;

        idx_rhs = rhs.rank() - i;
        this->eff_rhs[idx_res] = i <= rhs.rank() ? rhs.strides[idx_rhs] : 0;

        // make sure eff_res[idx] is 1 instead of 0 if
        // res.shape[idx] is 1 for traversing in flexible function
        if (this->eff_res[idx_res] == 0 && res.shape[idx_res] == 1) {
            this->eff_res[idx_res] = 1;
        }
    }

    this->res_ends.resize(res.rank());
    for (std::size_t i = 0; i < res.rank(); i++) {
        this->res_ends[i] =
            static_cast<std::size_t>(res.shape[i]) * this->eff_res[i];
    }
}

} // namespace kaad::functions
