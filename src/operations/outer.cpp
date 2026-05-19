#include "outer.hpp"

#include "kaad/graph/internal/inode.hpp"
#include "kaad/tensor/internal/tensor.hpp"
#include "kaad/tensor/internal/tensor_types.hpp"

#include <algorithm>
#include <array>

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

    const Tensor &lhs = inputs[0]->value;
    const Tensor &rhs = inputs[1]->value;
    const Tensor &res = result->value;

    eff_lhs.resize(res.rank());
    eff_rhs.resize(res.rank());
    eff_res.resize(res.rank());

    std::ranges::copy(lhs.strides, eff_lhs.begin());

    std::ranges::copy(rhs.strides, eff_rhs.begin() + lhs.rank());

    std::ranges::copy(res.strides, eff_res.begin());
}

} // namespace kaad::operations
