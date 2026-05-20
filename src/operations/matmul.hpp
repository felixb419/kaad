#pragma once

#include "../operations/operation_concept.hpp"
#include "kaad/graph/internal/inode.hpp"
#include "kaad/tensor/internal/tensor_types.hpp"

#include <array>
#include <cstddef>
#include <kaad/max_rank.hpp>
#include <kaad/scalar.hpp>
#include <kaad/static_vector.hpp>
#include <kaad/tensor/tensor_view.hpp>
#include <utility>

namespace kaad::operations {

/// @internal
struct Matmul {

    static constexpr std::size_t ARITY = 2;

    static constexpr const char *OPERATION_NAME = "matmul";

    /**
     * @brief Broadcasts @p lhs and @p rhs according to matrix
     * multiplication, additional axes will be treated as batch
     * axes.
     * @throws kaad::BroadcastError If @p lhs and @p rhs are not compatible.
     */
    static Shape make_res_shape(ShapeView lhs, ShapeView rhs);

    /// @copydoc make_res_shape
    static Shape make_res_shape(std::array<INode *, 2> inputs);

    struct ForwardParams {

        const Scalar *lhs_begin;
        const Scalar *rhs_begin;
        Scalar *res_begin;

        Stride lhs_col_step;  ///< Step size between columns of lhs.
        Stride rhs_row_step;  ///< Step size between rows of rhs.
        Extent extent_shared; ///< Length of the shared axis of lhs and rhs.

        Strides eff_lhs; ///< Broadcasted strides for lhs.
        Strides eff_rhs; ///< Broadcasted strides for rhs.
        Strides eff_res; ///< Broadcasted strides for res.

        Shape res_broadcast; ///< Broadcast of lhs and rhs (might differ
                             ///< from res.shape() in backward pass)

        ForwardParams() = default;

        ForwardParams(const Tensor &lhs, const Tensor &rhs, const Tensor &res);

        ForwardParams(std::array<INode *, 2> inputs, INode *result);
    };

    static void contract_leaf(const ForwardParams &params,
                              std::size_t lhs_offset, std::size_t rhs_offset,
                              std::size_t res_idx) {

        for (std::size_t i = 0; i < params.extent_shared; i++) {

            const std::size_t LHS_IDX = lhs_offset + (i * params.lhs_col_step);
            const std::size_t RHS_IDX = rhs_offset + (i * params.rhs_row_step);

            params.res_begin[res_idx] +=
                params.lhs_begin[LHS_IDX] * params.rhs_begin[RHS_IDX];
        }
    }

    template <std::size_t res_rank, std::size_t axis = 0>
        requires(res_rank > 0 && axis < res_rank)
    static void forward_walk(const ForwardParams &params,
                             std::size_t lhs_offset, std::size_t rhs_offset,
                             std::size_t res_offset) {

        for (std::size_t i = 0; i < params.res_broadcast[axis]; i++) {

            const std::size_t LHS_IDX = lhs_offset + (i * params.eff_lhs[axis]);
            const std::size_t RHS_IDX = rhs_offset + (i * params.eff_rhs[axis]);
            const std::size_t RES_IDX = res_offset + (i * params.eff_res[axis]);

            if constexpr (axis >= res_rank - 1) {

                contract_leaf(params, LHS_IDX, RHS_IDX, RES_IDX);

            } else {

                forward_walk<res_rank, axis + 1>(params, LHS_IDX, RHS_IDX,
                                                 RES_IDX);
            }
        }
    }

    using forward_fn = void (*)(const ForwardParams &params);

    template <std::size_t res_rank>
        requires(res_rank > 0)
    static void forward(const ForwardParams &params) noexcept {

        forward_walk<res_rank, 0>(params, 0, 0, 0);
    }

    struct BackwardParams {
        ForwardParams wrt_lhs;
        ForwardParams wrt_rhs;

        BackwardParams() = default;

        BackwardParams(std::array<INode *, 2> inputs, INode *result);
    };

    /// Signatrue of dispatched @ref backward
    using backward_fn = void (*)(const BackwardParams &params);

    /// Accumulates the gradients w.r.t. the inputs.
    template <std::size_t res_rank>
        requires(res_rank > 0)
    static void backward(const BackwardParams &params) noexcept {

        // d_res * rhs^T = d_lhs
        forward<res_rank>(params.wrt_lhs);

        // lhs^T * d_res = d_rhs
        forward<res_rank>(params.wrt_rhs);
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

static_assert(Operation<Matmul>);

} // namespace kaad::operations
