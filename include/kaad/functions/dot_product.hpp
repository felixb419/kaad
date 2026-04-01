#pragma once

#include <kaad/enums.hpp>               // for ScalarOrder
#include <kaad/scalar.hpp>              // for Scalar
#include <kaad/tensor/tensor_types.hpp> // for ShapeView, Shape

namespace kaad::functions {

struct DotProduct {

    /// @brief Checks if @p lhs and @p rhs are compatible for a dot product,
    /// throws kaad::BroadcastError otherwise.
    static void broadcast(ShapeView lhs, ShapeView rhs);

    using primal_fn = void (*)(const Scalar *lhs, const Scalar *rhs,
                               Scalar *res, const Scalar *lhs_end);

    /**
     * @brief Computes the dot product of @p lhs and @p rhs into @p res.
     * @ingroup binary_primal_functions
     * @pre @p lhs and @p rhs are rank 0 or 1 (specified by @tp S) and @p res is
     * rank-0.
     * @tparam S Indicator of which input is scalar.
     * @param[in] lhs Pointer to the start of rank-0/1 tensor.
     * @param[in] rhs Pointer to the start of rank-0/1 tensor
     * @param[out] res Pointer to rank-0 tensor.
     * @param end Pointer to the end of the non-scalar input (end of @p lhs if
     * none are scalar).
     */
    template <ScalarOrder S = NONE_SCALAR>
    static void primal(const Scalar *lhs, const Scalar *rhs, Scalar *res,
                       const Scalar *end) noexcept;

    using adjoint_fn = void (*)(const Scalar *lhs, Scalar *d_lhs,
                                const Scalar *rhs, Scalar *d_rhs,
                                const Scalar *d_res, const Scalar *end);

    /**
     * @brief Accumulates the gradient of the dot-product of @p lhs and @p rhs.
     * @ingroup binary_adjoint_functions
     * @pre @p lhs and @p rhs are rank 0 or 1 (specified by @tp S) and @p res is
     * rank-0.
     * @pre Every operand must have the same shape as their gradient.
     * @tparam S Indicator of which input is scalar.
     * @param[in] lhs Pointer to the start of rank-0/1 tensor.
     * @param[out] d_lhs Pointer to the start of the gradient w.r.t. @p lhs.
     * @param[in] rhs Pointer to the start of rank-0/1 tensor.
     * @param[out] d_rhs Pointer to the start of the gradient w.r.t. @p rhs.
     * @param[in] res Pointer to the start of rank-0 tensor.
     * @param[in] d_res Pointer to the start of the gradient w.r.t. @p res.
     * @param end Pointer to the end of the non-scalar input (end of @p lhs if
     * none are scalar).
     */
    template <ScalarOrder S = NONE_SCALAR>
    static void adjoint(const Scalar *lhs, Scalar *d_lhs, const Scalar *rhs,
                        Scalar *d_rhs, const Scalar *d_res,
                        const Scalar *end) noexcept;
};

} // namespace kaad::functions
