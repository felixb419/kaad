#pragma once

#include "../operations/operation_concept.hpp"
#include <array>
#include <concepts>
#include <cstddef>
#include <kaad/graph/internal/inode.hpp>
#include <kaad/max_rank.hpp>
#include <kaad/operators/internal/kernels.hpp>
#include <kaad/scalar.hpp>
#include <kaad/tensor/internal/tensor_types.hpp>
#include <utility>

namespace kaad::operations {

template <class Policy>
concept FlexiblePolicy =
    requires(std::array<INode *, 2> inputs, INode *result, Strides &eff_lhs,
             Strides &eff_rhs, Strides &eff_res) {
        { Policy::make_res_shape(inputs) } -> std::same_as<Shape>;

        Policy::init_strides(inputs, result, eff_lhs, eff_rhs, eff_res);
    };

template <kernels::Binary Kernel, FlexiblePolicy Policy> struct Strided {

    static constexpr std::size_t ARITY = 2;

    static constexpr const char *OPERATION_NAME = Kernel::OPERATION_NAME;

    /// @throws kaad::BroadcastError If shapes are incompatible according to
    /// numpy broadcasting rules.
    static Shape make_res_shape(std::array<INode *, 2> inputs) {
        return Policy::make_res_shape(inputs);
    }

    struct ForwardParams {
        const Scalar *lhs_begin;
        const Scalar *rhs_begin;
        Scalar *res_begin;

        Strides eff_lhs; ///< Broadcasted strides for lhs.
        Strides eff_rhs; ///< Broadcasted strides for rhs.
        Strides eff_res; ///< Broadcasted strides for res.

        Shape res_shape;

        ForwardParams() =
            default; ///< Needed because operations::Outer uses it.

        ForwardParams(std::array<INode *, 2> inputs, INode *result)
            : lhs_begin(inputs[0]->value().data()),
              rhs_begin(inputs[1]->value().data()),
              res_begin(result->value_mut().data()), eff_lhs(result->rank()),
              eff_rhs(result->rank()), eff_res(result->rank()),
              res_shape(result->shape()) {

            Policy::init_strides(inputs, result, this->eff_lhs, this->eff_rhs,
                                 this->eff_res);
        }
    };

    template <std::size_t res_rank, std::size_t axis = 0>
        requires(res_rank > 0 && axis < res_rank)
    static void forward_walk(
        const ForwardParams &params, std::size_t lhs_offset,
        std::size_t rhs_offset,
        std::size_t res_offset) noexcept(kernels::binary_noexcept<Kernel>()) {

        for (std::size_t i = 0; i < params.res_shape[axis]; i++) {

            const std::size_t LHS_IDX = lhs_offset + (i * params.eff_lhs[axis]);
            const std::size_t RHS_IDX = rhs_offset + (i * params.eff_rhs[axis]);
            const std::size_t RES_IDX = res_offset + (i * params.eff_res[axis]);

            if constexpr (axis >= res_rank - 1) {

                Kernel::op(params.lhs_begin[LHS_IDX], params.rhs_begin[RHS_IDX],
                           params.res_begin[RES_IDX]);

            } else {

                forward_walk<res_rank, axis + 1>(params, LHS_IDX, RHS_IDX,
                                                 RES_IDX);
            }
        }
    }

    using forward_fn = void (*)(const ForwardParams &params);

    template <std::size_t res_rank>
        requires(res_rank > 0)
    static void forward(const ForwardParams &params) noexcept(
        kernels::binary_noexcept<Kernel>()) {

        forward_walk<res_rank, 0>(params, 0, 0, 0);
    }

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

    template <std::size_t res_rank, std::size_t axis = 0>
        requires(res_rank > 0 && axis < res_rank)
    static void backward_walk(
        const BackwardParams &params, std::size_t lhs_offset,
        std::size_t rhs_offset,
        std::size_t res_offset) noexcept(kernels::binary_noexcept<Kernel>()) {

        for (std::size_t i = 0; i < params.res_shape[axis]; i++) {

            const std::size_t LHS_IDX = lhs_offset + (i * params.eff_lhs[axis]);
            const std::size_t RHS_IDX = rhs_offset + (i * params.eff_rhs[axis]);
            const std::size_t RES_IDX = res_offset + (i * params.eff_res[axis]);

            if constexpr (axis >= res_rank - 1) {

                Kernel::grad(
                    params.lhs_begin[LHS_IDX], params.d_lhs_begin[LHS_IDX],
                    params.rhs_begin[RHS_IDX], params.d_rhs_begin[RHS_IDX],
                    params.res_begin[RES_IDX], params.d_res_begin[RES_IDX]);

            } else {

                backward_walk<res_rank, axis + 1>(params, LHS_IDX, RHS_IDX,
                                                  RES_IDX);
            }
        }
    }

    using backward_fn = void (*)(const BackwardParams &params);

    template <std::size_t res_rank>
        requires(res_rank > 0)
    static void backward(const BackwardParams &params) noexcept(
        kernels::binary_noexcept<Kernel>()) {

        backward_walk<res_rank, 0>(params, 0, 0, 0);
    }

  private:
    // NOLINTBEGIN(readability-named-parameter)
    template <std::size_t... Is>
    static constexpr std::array<forward_fn, sizeof...(Is)>
    make_forward_dispatch_table(std::index_sequence<Is...>) {
        // +1 to avoid forward<0>
        return {&forward<Is + 1>...};
    }

    template <std::size_t... Is>
    static constexpr std::array<backward_fn, sizeof...(Is)>
    make_backward_dispatch_table(std::index_sequence<Is...>) {
        // +1 to avoid forward<0>
        return {&backward<Is + 1>...};
    }
    // NOLINTEND(readability-named-parameter)

  public:
    struct Dispatch {
        forward_fn forward;
        backward_fn backward;
    };

    static Dispatch dispatch([[maybe_unused]] std::array<INode *, 2> inputs,
                             INode *result) {
        // -1 because of +1 in make table function
        return {
            .forward = make_forward_dispatch_table(
                std::make_index_sequence<KAAD_MAX_RANK>())[result->rank() - 1],
            .backward = make_backward_dispatch_table(
                std::make_index_sequence<KAAD_MAX_RANK>())[result->rank() - 1]};
    }
};

} // namespace kaad::operations
