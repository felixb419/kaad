#include "dot_product.hpp"

#include "kaad/graph/internal/inode.hpp"
#include "kaad/tensor/internal/tensor_types.hpp"

#include <algorithm>
#include <array>
#include <kaad/enums.hpp>
#include <kaad/exceptions.hpp>
#include <kaad/scalar.hpp>

namespace kaad::operations {

Shape DotProduct::make_res_shape(std::array<INode *, 2> inputs) {

    ShapeView lhs = inputs[0]->shape();
    ShapeView rhs = inputs[1]->shape();

    bool rank_greater_1 = lhs.size() > 1 || rhs.size() > 1;
    bool different_shapes = !std::ranges::equal(lhs, rhs);
    bool none_scalar = !lhs.empty() && !rhs.empty();

    if (rank_greater_1 || (none_scalar && different_shapes)) {
        throw ShapeError(
            "incompatible tensor shapes for dot product, A.shape=" +
            to_string(lhs) + ", B.shape=" + to_string(rhs));
    }

    return SCALAR_SHAPE;
}

template <>
void DotProduct::forward<NONE_SCALAR>(const ForwardParams &params) noexcept {

    const Scalar *lhs = params.lhs_begin;
    const Scalar *rhs = params.rhs_begin;

    for (; lhs != params.lhs_end; lhs++, rhs++) {

        *params.res_begin += *lhs * (*rhs);
    }
}

template <>
void DotProduct::forward<LHS_IS_SCALAR>(const ForwardParams &params) noexcept {

    const Scalar *rhs = params.rhs_begin;

    for (; rhs != params.rhs_end; rhs++) {
        *params.res_begin += (*rhs) * (*params.lhs_begin);
    }
}

template <>
void DotProduct::forward<RHS_IS_SCALAR>(const ForwardParams &params) noexcept {

    const Scalar *lhs = params.lhs_begin;

    for (; lhs != params.lhs_end; lhs++) {
        *params.res_begin += (*lhs) * (*params.rhs_begin);
    }
}

template <>
void DotProduct::backward<NONE_SCALAR>(const BackwardParams &params) noexcept {

    const Scalar *lhs = params.lhs_begin;
    Scalar *d_lhs = params.d_lhs_begin;
    const Scalar *rhs = params.rhs_begin;
    Scalar *d_rhs = params.d_rhs_begin;

    for (; lhs != params.lhs_end; lhs++, d_lhs++, rhs++, d_rhs++) {

        *d_lhs += *params.d_res_begin * (*rhs);
        *d_rhs += *params.d_res_begin * (*lhs);
    }
}

template <>
void DotProduct::backward<LHS_IS_SCALAR>(
    const BackwardParams &params) noexcept {

    const Scalar *rhs = params.rhs_begin;
    Scalar *d_rhs = params.d_rhs_begin;

    for (; rhs != params.rhs_end; rhs++, d_rhs++) {

        *d_rhs += (*params.d_res_begin) * (*params.lhs_begin);
        *params.d_lhs_begin += (*params.d_res_begin) * (*rhs);
    }
}

template <>
void DotProduct::backward<RHS_IS_SCALAR>(
    const BackwardParams &params) noexcept {

    const Scalar *lhs = params.lhs_begin;
    Scalar *d_lhs = params.d_lhs_begin;

    for (; lhs != params.lhs_end; lhs++, d_lhs++) {

        *params.d_rhs_begin += (*params.d_res_begin) * (*lhs);
        *d_lhs += (*params.d_res_begin) * (*params.rhs_begin);
    }
}

DotProduct::Dispatch DotProduct::dispatch(std::array<INode *, 2> inputs,
                                          [[maybe_unused]] INode *result) {

    bool lhs_scalar = inputs[0]->value.scalar();
    bool rhs_scalar = inputs[1]->value.scalar();

    if (lhs_scalar) {
        return {.forward = forward<LHS_IS_SCALAR>,
                .backward = backward<LHS_IS_SCALAR>};
    }

    if (rhs_scalar) {
        return {.forward = forward<RHS_IS_SCALAR>,
                .backward = backward<RHS_IS_SCALAR>};
    }

    return {.forward = forward<NONE_SCALAR>, .backward = backward<NONE_SCALAR>};
}

} // namespace kaad::operations
