#include <kaad/functions/dot_product.hpp> // for DotProduct

#include <algorithm>                    // for __equal_fn, equal
#include <kaad/enums.hpp>               // for ScalarOrder
#include <kaad/exceptions.hpp>          // for BroadcastError, to_string
#include <kaad/scalar.hpp>              // for Scalar
#include <kaad/tensor/tensor_types.hpp> // for ShapeView
#include <string>                       // for allocator, char_traits, oper...

namespace kaad::functions {

void DotProduct::broadcast(ShapeView lhs, ShapeView rhs) {

    bool rank_greater_1 = lhs.size() > 1 || rhs.size() > 1;
    bool different_shapes = !std::ranges::equal(lhs, rhs);
    bool none_scalar = !lhs.empty() && !rhs.empty();

    if (rank_greater_1 || (none_scalar && different_shapes)) {
        throw BroadcastError(
            "incompatible tensor shapes for dot product, A.shape=" +
            to_string(lhs) + ", B.shape=" + to_string(rhs));
    }
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
