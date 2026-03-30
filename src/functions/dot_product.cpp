#include <kaad/functions/dot_product.hpp> // for DotProduct

#include <algorithm>                    // for __equal_fn, equal
#include <kaad/enums.hpp>               // for ScalarOrder
#include <kaad/scalar.hpp>              // for Scalar
#include <kaad/static_vector.hpp>       // for StaticVector
#include <kaad/tensor/tensor_types.hpp> // for ShapeView, Shape

namespace kaad::functions {

bool DotProduct::broadcast(ShapeView lhs, ShapeView rhs, Shape &new_shape) {
    // if any above rank-1 -> fails
    if (lhs.size() > 1 || rhs.size() > 1) {
        return false;
    }

    // if both non-scalar and shapes dont match -> fails
    if (!lhs.empty() && !rhs.empty() && !std::ranges::equal(lhs, rhs)) {
        return false;
    }

    new_shape = {1};
    return true;
}

template <>
void DotProduct::primal<NONE_SCALAR>(const Scalar *lhs, const Scalar *rhs,
                                     Scalar *res, const Scalar *end) noexcept {

    for (; lhs != end; lhs++, rhs++) {
        *res += *lhs * (*rhs);
    }
}

template <>
void DotProduct::primal<LHS_IS_SCALAR>(const Scalar *lhs, const Scalar *rhs,
                                       Scalar *res,
                                       const Scalar *end) noexcept {

    for (; rhs != end; rhs++) {
        *res += (*rhs) * (*lhs);
    }
}

template <>
void DotProduct::primal<RHS_IS_SCALAR>(const Scalar *lhs, const Scalar *rhs,
                                       Scalar *res,
                                       const Scalar *end) noexcept {

    for (; lhs != end; lhs++) {
        *res += (*lhs) * (*rhs);
    }
}

template <>
void DotProduct::adjoint<NONE_SCALAR>(const Scalar *lhs, Scalar *d_lhs,
                                      const Scalar *rhs, Scalar *d_rhs,
                                      const Scalar *d_res,
                                      const Scalar *lhs_end) noexcept {

    for (; lhs != lhs_end; lhs++, d_lhs++, rhs++, d_rhs++) {
        *d_lhs += *d_res * (*rhs);
        *d_rhs += *d_res * (*lhs);
    }
}

template <>
void DotProduct::adjoint<LHS_IS_SCALAR>(const Scalar *lhs, Scalar *d_lhs,
                                        const Scalar *rhs, Scalar *d_rhs,
                                        const Scalar *d_res,
                                        const Scalar *end) noexcept {

    for (; rhs != end; rhs++, d_rhs++) {
        *d_rhs += (*d_res) * (*lhs);
        *d_lhs += (*d_res) * (*rhs);
    }
}

template <>
void DotProduct::adjoint<RHS_IS_SCALAR>(const Scalar *lhs, Scalar *d_lhs,
                                        const Scalar *rhs, Scalar *d_rhs,
                                        const Scalar *d_res,
                                        const Scalar *end) noexcept {

    for (; lhs != end; lhs++, d_lhs++) {
        *d_rhs += (*d_res) * (*lhs);
        *d_lhs += (*d_res) * (*rhs);
    }
}

} // namespace kaad::functions
