#pragma once

#include <array>                        // for array
#include <kaad/operations/kernels.hpp>  // for Mul
#include <kaad/operations/strided.hpp>  // for Flexible
#include <kaad/scalar.hpp>              // for Scalar
#include <kaad/tensor/tensor_types.hpp> // for Strides, Shape

namespace kaad {
class INode;
}

namespace kaad::operations {

struct OuterProductPolicy {

    static Shape make_res_shape(std::array<INode *, 2> inputs);

    static void init_strides(std::array<INode *, 2> inputs, INode *result,
                             Strides &eff_lhs, Strides &eff_rhs,
                             Strides &eff_res);
};

using OuterProduct = Strided<kernels::Mul<Scalar>, OuterProductPolicy>;

} // namespace kaad::operations
