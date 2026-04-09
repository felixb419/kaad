#pragma once

#include "kaad/functions/kernels.hpp" // for binary_kernel_class, bin_kerne...
#include <kaad/enums.hpp>             // for ScalarOrder

namespace kaad::functions {

struct Pointwise {

    struct Binary {

        template <binary_kernel_class Kernel>
        using primal_fn = void (*)(const typename Kernel::value_type *lhs,
                                   const typename Kernel::value_type *rhs,
                                   typename Kernel::value_type *res,
                                   const typename Kernel::value_type *res_end);

        /**
         * @brief Applies Op to @p lhssand @p rhs.
         * @ingroup binary_primal_functions
         * @pre @p lhs and @p rhs have the same shape (or are a scalar if
         * specified by @tp S).
         * @tparam Kernel Binary kernel type with a static `op` function.
         * @tparam S Selects scalar handling: no scalar, left scalar, or right
         * scalar.
         * @param[in] lhs Pointer to the start of tensor.
         * @param[in] rhs Pointer to the start of tensor.
         * @param[out] res Pointer to the start of tensor.
         * @param res_end Pointer to the end of @p res.
         */
        template <binary_kernel_class Kernel, ScalarOrder S = NONE_SCALAR>
            requires(S == NONE_SCALAR || S == LHS_IS_SCALAR ||
                     S == RHS_IS_SCALAR)
        static void
        primal(const Kernel::value_type *lhs, const Kernel::value_type *rhs,
               Kernel::value_type *res,
               const Kernel::value_type
                   *res_end) noexcept(bin_kernel_noexcept<Kernel>()) {
            if constexpr (S == NONE_SCALAR) {

                for (; res != res_end; lhs++, rhs++, res++) {
                    Kernel::op(*lhs, *rhs, *res);
                }

            } else if constexpr (S == LHS_IS_SCALAR) {

                for (; res != res_end; rhs++, res++) {
                    Kernel::op(*lhs, *rhs, *res);
                }

            } else if constexpr (S == RHS_IS_SCALAR) {

                for (; res != res_end; lhs++, res++) {
                    Kernel::op(*lhs, *rhs, *res);
                }
            }
        }

        static int test_fn(int par);

        template <binary_kernel_class Kernel>
        using adjoint_fn = void (*)(const typename Kernel::value_type *lhs,
                                    typename Kernel::value_type *d_lhs,
                                    const typename Kernel::value_type *rhs,
                                    typename Kernel::value_type *d_rhs,
                                    const typename Kernel::value_type *res,
                                    const typename Kernel::value_type *d_res,
                                    const typename Kernel::value_type *end);

        /**
         * @brief Accumulates the gradient of Op, @p lhs , @p rhs .
         * @ingroup binary_adjoint_functions
         * @pre @p lhs, @p rhs and @p res have the same shape (or are a scalar
         * if specified by @tp S).
         * @pre Every operand must have the same shape as their gradient.
         * @tparam Kernel A struct containing a static binary funcion ('Grad').
         * @tparam S Selects scalar handling: no scalar, left scalar, or right
         * scalar.
         * @param[in] lhs Pointer to the start of tensor.
         * @param[out] d_lhs Pointer to the start of the gradient w.r.t. @p lhs.
         * @param[in] rhs Pointer to the start of tensor.
         * @param[out] d_rhs Pointer to the start of the gradient w.r.t. @p rhs.
         * @param[in] res Pointer to the start of tensor.
         * @param[in] d_res Pointer to the start of the gradient w.r.t. @p res.
         * @param res_end Pointer to the end of @p res.
         */
        template <binary_kernel_class Kernel, ScalarOrder S = NONE_SCALAR>
            requires(S == NONE_SCALAR || S == LHS_IS_SCALAR ||
                     S == RHS_IS_SCALAR)
        static void
        adjoint(const Kernel::value_type *lhs, Kernel::value_type *d_lhs,
                const Kernel::value_type *rhs, Kernel::value_type *d_rhs,
                const Kernel::value_type *res, const Kernel::value_type *d_res,
                const Kernel::value_type
                    *res_end) noexcept(bin_kernel_noexcept<Kernel>()) {

            if constexpr (S == NONE_SCALAR) {

                for (; res != res_end;
                     lhs++, d_lhs++, rhs++, d_rhs++, res++, d_res++) {
                    Kernel::grad(*lhs, *d_lhs, *rhs, *d_rhs, *res, *d_res);
                }

            } else if constexpr (S == LHS_IS_SCALAR) {

                for (; res != res_end; rhs++, d_rhs++, res++, d_res++) {
                    Kernel::grad(*lhs, *d_lhs, *rhs, *d_rhs, *res, *d_res);
                }

            } else if constexpr (S == RHS_IS_SCALAR) {

                for (; res != res_end; lhs++, d_lhs++, res++, d_res++) {
                    Kernel::grad(*lhs, *d_lhs, *rhs, *d_rhs, *res, *d_res);
                }
            }
        }
    };

    struct Unary {

        template <unary_kernel_class Kernel>
        using primal_fn = void (*)(const typename Kernel::value_type *inp,
                                   typename Kernel::value_type *res,
                                   const typename Kernel::value_type *res_end);

        /**
         * @brief Applies a unary operation to @p inp .
         * @ingroup unary_primal_functions
         * @tparam Kernel A struct containing a static unary function ('Op').
         * @param[in] inp Pointer to the start of tensor.
         * @param[out] res Pointer to the start of tensor
         * @param res_end Pointer to the end of @p res.
         * @param op Instance of the callable class.
         */
        template <unary_kernel_class Kernel>
        void static primal(const Kernel::value_type *inp,
                           Kernel::value_type *res,
                           const Kernel::value_type *
                               res_end) noexcept(un_kernel_noexcept<Kernel>()) {
            for (; res != res_end; inp++, res++) {
                Kernel::op(*inp, *res);
            }
        }

        /**
         * @brief Applies a unary operation to @p inp .
         * @ingroup unary_primal_functions
         * @tparam Kernel A struct containing a static unary function ('Op').
         * @param[in] inp Pointer to the start of tensor.
         * @param[out] res Pointer to rank-0 tensor.
         * @param inp_end Pointer to the end of @p inp.
         */
        template <unary_kernel_class Kernel>
        void static primal_scalar_res(
            const Kernel::value_type *inp, Kernel::value_type *res,
            const Kernel::value_type
                *inp_end) noexcept(un_kernel_noexcept<Kernel>()) {
            for (; inp != inp_end; inp++) {
                Kernel::op(*inp, *res);
            }
        }

        template <unary_kernel_class Kernel>
        using adjoint_fn = void (*)(const typename Kernel::value_type *inp,
                                    typename Kernel::value_type *d_inp,
                                    const typename Kernel::value_type *res,
                                    const typename Kernel::value_type *d_res,
                                    const typename Kernel::value_type *end);

        /**
         * @brief Accumulates the gradient of Op in @p inp .
         * @ingroup unary_adjoint_functions
         * @pre @p inp and @p res have the same shape.
         * @pre Every operand must have the same shape as their gradient.
         * @tparam Kernel A struct containing a static binary funcion ('Grad').
         * @param[in] inp Pointer to the start of tensor.
         * @param[out] d_inp Pointer to the start of the gradient w.r.t. @p inp.
         * @param[in] res Pointer to the start of tensor.
         * @param[in] d_res Pointer to the start of the gradient w.r.t. @p res.
         * @param res_end Pointer to the end of @p res.
         */
        template <unary_kernel_class Kernel>
        static void
        adjoint(const Kernel::value_type *inp, Kernel::value_type *d_inp,
                const Kernel::value_type *res, const Kernel::value_type *d_res,
                const Kernel::value_type
                    *res_end) noexcept(un_kernel_noexcept<Kernel>()) {
            for (; res != res_end; inp++, d_inp++, res++, d_res++) {
                Kernel::grad(*inp, *d_inp, *res, *d_res);
            }
        }

        /**
         * @brief Accumulates the gradient of Op in @p inp .
         * @ingroup unary_adjoint_functions
         * @pre @p res is 0-rank.
         * @pre Every operand must have the same shape as their gradient.
         * @tparam Kernel A struct containing a static binary funcion ('Grad').
         * @param[in] inp Pointer to the start of tensor.
         * @param[out] d_inp Pointer to the start of the gradient w.r.t. @p inp.
         * @param[in] res Pointer to the start of 0-rank tensor.
         * @param[in] d_res Pointer to the start of the gradient w.r.t. @p res.
         * @param inp_end Pointer to the end of @p inp.
         */
        template <unary_kernel_class Kernel>
        static void adjoint_scalar_res(
            const Kernel::value_type *inp, Kernel::value_type *d_inp,
            const Kernel::value_type *res, const Kernel::value_type *d_res,
            const Kernel::value_type
                *inp_end) noexcept(un_kernel_noexcept<Kernel>()) {
            for (; inp != inp_end; inp++, d_inp++) {
                Kernel::grad(*inp, *d_inp, *res, *d_res);
            }
        }
    };
};

} // namespace kaad::functions
