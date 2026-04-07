#pragma once

#include <array>                        // for array
#include <cstddef>                      // for size_t
#include <kaad/functions/kernels.hpp>   // for binary_kernel_class, bin_ker...
#include <kaad/max_rank.hpp>            // for KAAD_MAX_RANK
#include <kaad/static_vector.hpp>       // for StaticVector
#include <kaad/tensor/tensor_types.hpp> // for Strides, ShapeView, Shape
#include <kaad/tensor/tensor_view.hpp>  // for TensorViewConst
#include <utility>                      // for index_sequence, make_index_s...

namespace kaad::functions {

struct Flexible {

    /// @brief Broadcasts @p lhs and @p rhs, will throw BroadcastError if @p lhs
    /// and @p rhs are not compatible.
    static Shape broadcast(ShapeView lhs, ShapeView rhs);

    struct Metadata {

        Strides eff_lhs; ///< Broadcasted strides for lhs.
        Strides eff_rhs; ///< Broadcasted strides for rhs.
        Strides eff_res; ///< Broadcasted strides for res.

        StaticVector<std::size_t> res_ends;

        Metadata() = default;

        Metadata(TensorViewConst lhs, TensorViewConst rhs, TensorViewConst res);
    };

    template <binary_kernel_class Kernel>
    using primal_fn = void (*)(const typename Kernel::value_type *lhs,
                               const typename Kernel::value_type *rhs,
                               typename Kernel::value_type *res,
                               Metadata mdata);

    /**
     * @brief Applies Op to @p lhssand @p rhs.
     * @ingroup binary_primal_functions
     * @pre @p res shape is the result of broadcasting @p lhs and @p rhs.
     * @tparam Kernel A struct containing a static binary function ('Op').
     * @tparam rank Rank of the out
     * @param[in] lhs Pointer to the start of tensor.
     * @param[in] rhs Pointer to the start of tensor.
     * @param[out] res Pointer to the start of tensor.
     * @param mdata Metadata needed for traversal.
     */
    template <binary_kernel_class Kernel, std::size_t res_rank,
              std::size_t dim = 0>
        requires(res_rank > 0 && dim < res_rank)
    static void primal(const Kernel::value_type *lhs,
                       const Kernel::value_type *rhs, Kernel::value_type *res,
                       Metadata mdata) noexcept(bin_kernel_noexcept<Kernel>()) {

        const typename Kernel::value_type *end = res + mdata.res_ends[dim];
        if constexpr (dim >= res_rank - 1) {

            for (; res != end; lhs += mdata.eff_lhs[dim],
                               rhs += mdata.eff_rhs[dim],
                               res += mdata.eff_res[dim]) {

                Kernel::op(*lhs, *rhs, *res);
            }
        } else {

            for (; res < end; lhs += mdata.eff_lhs[dim],
                              rhs += mdata.eff_rhs[dim],
                              res += mdata.eff_res[dim]) {

                primal<Kernel, res_rank, dim + 1>(lhs, rhs, res, mdata);
            }
        }
    }

    template <binary_kernel_class Kernel>
    using adjoint_fn = void (*)(const typename Kernel::value_type *lhs,
                                typename Kernel::value_type *d_lhs,
                                const typename Kernel::value_type *rhs,
                                typename Kernel::value_type *d_rhs,
                                const typename Kernel::value_type *res,
                                const typename Kernel::value_type *d_res,
                                Metadata mdata);

    /**
     * @brief Accumulates the gradient of Op, @p lhs , @p rhs .
     * @ingroup binary_adjoint_functions
     * @pre @p res shape is the result of broadcasting @p lhs and @p rhs.
     * @pre Every operand must have the same shape as their gradient.
     * @tparam Kernel A struct containing a static binary funcion ('Grad').
     * @param[in] lhs Pointer to the start of tensor.
     * @param[out] d_lhs Pointer to the start of the gradient w.r.t. @p lhs.
     * @param[in] rhs Pointer to the start of tensor.
     * @param[out] d_rhs Pointer to the start of the gradient w.r.t. @p rhs.
     * @param[in] res Pointer to the start of tensor.
     * @param[in] d_res Pointer to the start of the gradient w.r.t. @p res.
     * @param strides_lhs Stride array of @p lhs.
     * @param strides_rhs Stride array of @p rhs.
     * @param strides_res Stride array of @p res.
     * @param res_dim_offset Offset to the end of @p res per dimension.
     */
    template <binary_kernel_class Kernel, std::size_t res_rank,
              std::size_t dim = 0>
        requires(res_rank > 0 && dim < res_rank)
    static void
    adjoint(const Kernel::value_type *lhs, Kernel::value_type *d_lhs,
            const Kernel::value_type *rhs, Kernel::value_type *d_rhs,
            const Kernel::value_type *res, const Kernel::value_type *d_res,
            Metadata mdata) noexcept(bin_kernel_noexcept<Kernel>()) {

        const typename Kernel::value_type *end = res + mdata.res_ends[dim];

        if constexpr (dim >= res_rank - 1) {

            for (; res != end;
                 lhs += mdata.eff_lhs[dim], rhs += mdata.eff_rhs[dim],
                 res += mdata.eff_res[dim], d_lhs += mdata.eff_lhs[dim],
                 d_rhs += mdata.eff_rhs[dim], d_res += mdata.eff_res[dim]) {

                Kernel::grad(*lhs, *d_lhs, *rhs, *d_rhs, *res, *d_res);
            }
        } else {

            for (; res != end;
                 lhs += mdata.eff_lhs[dim], rhs += mdata.eff_rhs[dim],
                 res += mdata.eff_res[dim], d_lhs += mdata.eff_lhs[dim],
                 d_rhs += mdata.eff_rhs[dim], d_res += mdata.eff_res[dim]) {

                adjoint<Kernel, res_rank, dim + 1>(lhs, d_lhs, rhs, d_rhs, res,
                                                   d_res, mdata);
            }
        }
    }

  private:
    // NOLINTBEGIN(readability-named-parameter)
    template <binary_kernel_class Kernel, std::size_t... Is>
    static constexpr std::array<primal_fn<Kernel>, sizeof...(Is)>
    make_primal_dispatch_table(std::index_sequence<Is...>) {
        // +1 to avoid primal<0>
        return {&primal<Kernel, Is + 1>...};
    }

    template <binary_kernel_class Kernel, std::size_t... Is>
    static constexpr std::array<adjoint_fn<Kernel>, sizeof...(Is)>
    make_adjoint_dispatch_table(std::index_sequence<Is...>) {
        // +1 to avoid primal<0>
        return {&adjoint<Kernel, Is + 1>...};
    }
    // NOLINTEND(readability-named-parameter)

  public:
    template <binary_kernel_class Kernel> struct Dispatch {
        primal_fn<Kernel> primal;
        adjoint_fn<Kernel> adjoint;
    };

    template <binary_kernel_class Kernel>
    static Dispatch<Kernel> dispatch(std::size_t res_rank) {
        // -1 because of +1 in make table function
        return {.primal = make_primal_dispatch_table<Kernel>(
                    std::make_index_sequence<KAAD_MAX_RANK>())[res_rank - 1],
                .adjoint = make_adjoint_dispatch_table<Kernel>(
                    std::make_index_sequence<KAAD_MAX_RANK>())[res_rank - 1]};
    }
};

} // namespace kaad::functions
