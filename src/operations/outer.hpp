#pragma once

#include "strided.hpp"

#include <array>
#include <kaad/operators/internal/kernels.hpp>
#include <kaad/scalar.hpp>
#include <kaad/tensor/internal/tensor_types.hpp>

namespace kaad {
struct INode;
}

namespace kaad::operations {

struct OuterProductPolicy {

    static Shape make_res_shape(std::array<INode *, 2> inputs);

    static void init_strides(std::array<INode *, 2> inputs, INode *result,
                             Strides &eff_lhs, Strides &eff_rhs,
                             Strides &eff_res);
};

using OuterProduct = Strided<kernels::Mul<Scalar>, OuterProductPolicy>;

static_assert(Operation<OuterProduct>);

} // namespace kaad::operations
