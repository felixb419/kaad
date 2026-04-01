#pragma once

#include <array>                        // for array
#include <cstddef>                      // for size_t
#include <kaad/max_rank.hpp>            // for KAAD_MAX_RANK
#include <kaad/scalar.hpp>              // for Scalar
#include <kaad/tensor/tensor_types.hpp> // for Stride, Shape, ShapeView
#include <kaad/tensor/tensor_view.hpp>  // for TensorViewConst
#include <utility>                      // for make_index_sequence, index_s...

namespace kaad::functions {

/// @internal
struct Matmul {
  public:
    /**
     * @brief Broadcasts @p lhs and @p rhs according to matrix
     * multiplication, additional dimensions will be treated as batch
     * dimensions, will throw BroadcastError if @p lhs and @p rhs are not
     * compatible.
     */
    static Shape broadcast(ShapeView lhs, ShapeView rhs);

    struct Metadata {

        stride lhs_col_step; ///< Step size between columns of lhs.
        stride rhs_row_step; ///< Step size between rows of rhs.
        extent shared_dim;   ///< Length of the shared dimension of lhs and rhs.

        Strides eff_lhs; ///< Broadcasted strides for lhs.
        Strides eff_rhs; ///< Broadcasted strides for rhs.
        Strides eff_res; ///< Broadcasted strides for res.

        Shape res_broadcast; ///< Broadcast of lhs and rhs (might differ
                             ///< from res.shape() in backward pass)

        Metadata() = default;

        Metadata(TensorViewConst lhs, TensorViewConst rhs, TensorViewConst res);
    };

    /// Signatrue of @c primal.
    using primal_fn = void (*)(const Scalar *lhs, const Scalar *rhs,
                               Scalar *res, const Metadata &mdata);

    /**
     * @brief Computes batched matrix product of @p lhs and @p rhs into @p
     * res.
     * @ingroup binary_primal_functions
     * @pre @p lhs, @p rhs and @p res have compatible shapes.
     * @tparam res_rank Rank of @p res.
     * @tparam dim Idx of the dimension the function is currently working
     * on.
     * @param[in] lhs Pointer to the elements of lhs.
     * @param[in] rhs Pointer to the elements of rhs.
     * @param[out] res Pointer to the elements of res.
     * @param[in] mdata Metadata needed for traversal.
     */
    template <std::size_t res_rank, std::size_t dim = 0>
        requires(res_rank > 0 && dim < res_rank)
    static void primal(const Scalar *lhs, const Scalar *rhs, Scalar *res,
                       const Metadata &mdata) noexcept;

    /// Signatrue of @c adjoint.
    using adjoint_fn = void (*)(const Scalar *lhs, Scalar *d_lhs,
                                const Scalar *rhs, Scalar *d_rhs,
                                const Scalar *d_res, const Metadata &wrt_lhs,
                                const Metadata &wrt_rhs);

    /**
     * @brief Accumulates the gradient of Op, @p lhs , @p rhs .
     * @ingroup binary_adjoint_functions
     * @pre @p lhs, @p rhs and @p res have compatible shapes.
     * @pre Every operand must have the same shape as their gradient.
     * @tparam res_rank Rank of @p res.
     * @param[in] lhs Pointer to value elements.
     * @param[out] d_lhs Pointer to gradient elements.
     * @param[in] rhs Pointer to value elements.
     * @param[out] d_rhs Pointer to gradient elements.
     * @param[in] d_res Pointer to gradient elements.
     * @param[in] wrt_lhs Metadata for gradient w.r.t to @p lhs.
     * @param[in] wrt_rhs Metadata for gradient w.r.t to @p rhs.
     */
    template <std::size_t res_rank>
        requires(res_rank > 0)
    static void adjoint(const Scalar *lhs, Scalar *d_lhs, const Scalar *rhs,
                        Scalar *d_rhs, const Scalar *d_res,
                        const Metadata &wrt_lhs,
                        const Metadata &wrt_rhs) noexcept;

  private:
    // NOLINTBEGIN(readability-named-parameter)
    template <std::size_t... Is>
    static constexpr std::array<primal_fn, sizeof...(Is)>
    make_primal_dispatch_table(std::index_sequence<Is...>) {
        // +1 to avoid primal<0>
        return {&primal<Is + 1>...};
    }

    template <std::size_t... Is>
    static constexpr std::array<adjoint_fn, sizeof...(Is)>
    make_adjoint_dispatch_table(std::index_sequence<Is...>) {
        // +1 to avoid primal<0>
        return {&adjoint<Is + 1>...};
    }
    // NOLINTEND(readability-named-parameter)

  public:
    struct Dispatch {
        primal_fn primal;
        adjoint_fn adjoint;
    };

    static Dispatch dispatch(std::size_t res_rank) {
        // -1 because of +1 in make table function
        return {.primal = make_primal_dispatch_table(
                    std::make_index_sequence<KAAD_MAX_RANK>())[res_rank - 1],
                .adjoint = make_adjoint_dispatch_table(
                    std::make_index_sequence<KAAD_MAX_RANK>())[res_rank - 1]};
    }
};

template <std::size_t res_rank, std::size_t dim>
    requires(res_rank > 0 && dim < res_rank)
void Matmul::primal(const Scalar *lhs, const Scalar *rhs, Scalar *res,
                    const Metadata &mdata) noexcept {

    if constexpr (dim >= res_rank - 1) {
        for (std::size_t i = 0; i < mdata.res_broadcast[dim]; i++,
                         lhs += mdata.eff_lhs[dim], rhs += mdata.eff_rhs[dim],
                         res += mdata.eff_res[dim]) {

            const Scalar *row_lhs = lhs;
            const Scalar *col_rhs = rhs;

            for (std::size_t j = 0; j < mdata.shared_dim; j++,
                             row_lhs += mdata.lhs_col_step,
                             col_rhs += mdata.rhs_row_step) {
                *res += (*row_lhs) * (*col_rhs);
            }
        }
    } else {

        for (std::size_t i = 0; i < mdata.res_broadcast[dim]; i++,
                         lhs += mdata.eff_lhs[dim], rhs += mdata.eff_rhs[dim],
                         res += mdata.eff_res[dim]) {
            primal<res_rank, dim + 1>(lhs, rhs, res, mdata);
        }
    }
}

template <std::size_t res_rank>
    requires(res_rank > 0)
void Matmul::adjoint(const Scalar *lhs, Scalar *d_lhs, const Scalar *rhs,
                     Scalar *d_rhs, const Scalar *d_res,
                     const Metadata &wrt_lhs,
                     const Metadata &wrt_rhs) noexcept {
    // d_res * rhs^T = d_lhs
    primal<res_rank>(d_res, rhs, d_lhs, wrt_lhs);
    // primal<wrt_lhs.res_broadcast.size()>(d_res, rhs, d_lhs, wrt_lhs);

    // lhs^T * d_res = d_rhs
    primal<res_rank>(lhs, d_res, d_rhs, wrt_rhs);
}

} // namespace kaad::functions
