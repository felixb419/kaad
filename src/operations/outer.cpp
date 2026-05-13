#include "outer.hpp"

#include <algorithm>
#include <array>
#include <kaad/graph/internal/inode.hpp>
#include <kaad/tensor/internal/tensor_types.hpp>
#include <kaad/tensor/tensor_view.hpp>

namespace kaad::operations {

Shape OuterProductPolicy::make_res_shape(std::array<INode *, 2> inputs) {

    ShapeView lhs = inputs[0]->shape();
    ShapeView rhs = inputs[1]->shape();

    Shape res(lhs.size() + rhs.size());

    std::ranges::copy(lhs, res.begin());
    std::ranges::copy(rhs, res.begin() + lhs.size());

    return res;
}

void OuterProductPolicy::init_strides(std::array<INode *, 2> inputs,
                                      INode *result, Strides &eff_lhs,
                                      Strides &eff_rhs, Strides &eff_res) {

    TensorViewConst lhs = inputs[0]->value();
    TensorViewConst rhs = inputs[1]->value();
    TensorViewMut res = result->value_mut();

    eff_lhs.resize(res.rank());
    eff_rhs.resize(res.rank());
    eff_res.resize(res.rank());

    std::ranges::copy(lhs.strides, eff_lhs.begin());

    std::ranges::copy(rhs.strides, eff_rhs.begin() + lhs.rank());

    std::ranges::copy(res.strides, eff_res.begin());
}

} // namespace kaad::operations
