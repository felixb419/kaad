#pragma once

#include <array>                            // for array
#include <cstddef>                          // for size_t
#include <kaad/enums.hpp>                   // for ScalarOrder
#include <kaad/graph/inode.hpp>             // for INode
#include <kaad/graph/operation_concept.hpp> // for Operation
#include <kaad/scalar.hpp>                  // for Scalar
#include <kaad/tensor/tensor_types.hpp>     // for Shape

namespace kaad::operations {

struct DotProduct {

    static constexpr std::size_t ARITY = 2;

    static constexpr const char *OPERATION_NAME = "dot product";

    /// @note Throws BroadcastError if shapes differ or are not rank-1; scalar
    /// inputs are broadcast automatically.
    static Shape make_res_shape(std::array<INode *, 2> inputs);

    struct ForwardParams {

        const Scalar *lhs_begin;
        const Scalar *rhs_begin;
        Scalar *res_begin;

        // either lhs_end or rhs_end is used depending on ScalarOrder
        const Scalar *lhs_end;
        const Scalar *rhs_end;

        ForwardParams(std::array<INode *, 2> inputs, INode *result)
            : lhs_begin(inputs[0]->value().data()),
              rhs_begin(inputs[1]->value().data()),
              res_begin(result->value_mut().data()),
              lhs_end(inputs[0]->value().data() + inputs[0]->value().size()),
              rhs_end(inputs[1]->value().data() + inputs[1]->value().size()) {}
    };

    using forward_fn = void (*)(const ForwardParams &params);

    template <ScalarOrder S>
        requires(S == NONE_SCALAR || S == LHS_IS_SCALAR || S == RHS_IS_SCALAR)
    static void forward(const ForwardParams &params) noexcept;

    struct BackwardParams : ForwardParams {

        Scalar *d_lhs_begin;
        Scalar *d_rhs_begin;
        const Scalar *d_res_begin;

        BackwardParams(std::array<INode *, 2> inputs, INode *result)
            : ForwardParams(inputs, result),
              d_lhs_begin(inputs[0]->gradient_mut().data()),
              d_rhs_begin(inputs[1]->gradient_mut().data()),
              d_res_begin(result->gradient().data()) {}
    };

    using backward_fn = void (*)(const BackwardParams &params);

    template <ScalarOrder S>
        requires(S == NONE_SCALAR || S == LHS_IS_SCALAR || S == RHS_IS_SCALAR)
    static void backward(const BackwardParams &params) noexcept;

    struct Dispatch {
        forward_fn forward;
        backward_fn backward;
    };

    static Dispatch dispatch(std::array<INode *, 2> inputs,
                             [[maybe_unused]] INode *result);
};

static_assert(Operation<DotProduct>);

} // namespace kaad::operations
