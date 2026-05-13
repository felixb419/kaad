#pragma once

#include "strided.hpp"
#include <kaad/graph/internal/inode.hpp>
#include <kaad/tensor/internal/tensor_types.hpp>

namespace kaad::operations {

struct BroadcastPolicy {

    static Shape make_res_shape(ShapeView lhs, ShapeView rhs);

    static Shape make_res_shape(std::array<INode *, 2> inputs);

    static void init_strides(std::array<INode *, 2> inputs, INode *result,
                             Strides &eff_lhs, Strides &eff_rhs,
                             Strides &eff_res);
};

template <kernels::Binary Kernel>
using Broadcast = Strided<Kernel, BroadcastPolicy>;

static_assert(Operation<Broadcast<kernels::Add<Scalar>>>);

} // namespace kaad::operations
