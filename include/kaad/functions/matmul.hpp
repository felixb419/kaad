#pragma once

#include <array>                            // for array
#include <cstddef>                          // for size_t
#include <kaad/graph/inode.hpp>             // for INode
#include <kaad/graph/operation_concept.hpp> // for Operation
#include <kaad/max_rank.hpp>                // for KAAD_MAX_RANK
#include <kaad/scalar.hpp>                  // for Scalar
#include <kaad/static_vector.hpp>           // for StaticVector
#include <kaad/tensor/tensor_types.hpp>     // for Shape, Strides, ShapeView
#include <kaad/tensor/tensor_view.hpp>      // for TensorViewConst, TensorV...
#include <utility>                          // for make_index_sequence, ind...

namespace kaad::functions {

/// @internal
struct Matmul {

    static constexpr std::size_t ARITY = 2;

    static constexpr const char *OPERATION_NAME = "matmul";

    /**
     * @brief Broadcasts @p lhs and @p rhs according to matrix
     * multiplication, additional dimensions will be treated as batch
     * dimensions, will throw BroadcastError if @p lhs and @p rhs are not
     * compatible.
     */
    static Shape make_res_shape(ShapeView lhs, ShapeView rhs);

    /// @copydoc make_res_shape
    static Shape make_res_shape(std::array<INode *, 2> inputs);

    struct ForwardParams {

        const Scalar *lhs_begin;
        const Scalar *rhs_begin;
        Scalar *res_begin;

        stride lhs_col_step; ///< Step size between columns of lhs.
        stride rhs_row_step; ///< Step size between rows of rhs.
        extent shared_dim;   ///< Length of the shared dimension of lhs and rhs.

        Strides eff_lhs; ///< Broadcasted strides for lhs.
        Strides eff_rhs; ///< Broadcasted strides for rhs.
        Strides eff_res; ///< Broadcasted strides for res.

        Shape res_broadcast; ///< Broadcast of lhs and rhs (might differ
                             ///< from res.shape() in backward pass)

        ForwardParams() = default;

        ForwardParams(TensorViewConst lhs, TensorViewConst rhs,
                      TensorViewMut res);

        ForwardParams(std::array<INode *, 2> inputs, INode *result);
    };

    static void contract_leaf(const ForwardParams &params,
                              std::size_t lhs_offset, std::size_t rhs_offset,
                              std::size_t res_idx) {

        for (std::size_t i = 0; i < params.shared_dim; i++) {

            const std::size_t LHS_IDX = lhs_offset + (i * params.lhs_col_step);
            const std::size_t RHS_IDX = rhs_offset + (i * params.rhs_row_step);

            params.res_begin[res_idx] +=
                params.lhs_begin[LHS_IDX] * params.rhs_begin[RHS_IDX];
        }
    }

    template <std::size_t res_rank, std::size_t dim = 0>
        requires(res_rank > 0 && dim < res_rank)
    static void forward_walk(const ForwardParams &params,
                             std::size_t lhs_offset, std::size_t rhs_offset,
                             std::size_t res_offset) {

        for (std::size_t i = 0; i < params.res_broadcast[dim]; i++) {

            const std::size_t LHS_IDX = lhs_offset + (i * params.eff_lhs[dim]);
            const std::size_t RHS_IDX = rhs_offset + (i * params.eff_rhs[dim]);
            const std::size_t RES_IDX = res_offset + (i * params.eff_res[dim]);

            if constexpr (dim >= res_rank - 1) {

                contract_leaf(params, LHS_IDX, RHS_IDX, RES_IDX);

            } else {

                forward_walk<res_rank, dim + 1>(params, LHS_IDX, RHS_IDX,
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

} // namespace kaad::functions
