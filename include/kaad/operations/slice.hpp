#pragma once

#include <array>                            // for array
#include <cstddef>                          // for size_t
#include <kaad/graph/inode.hpp>             // for INode
#include <kaad/graph/operation_concept.hpp> // for Operation
#include <kaad/max_rank.hpp>                // for KAAD_MAX_RANK
#include <kaad/scalar.hpp>                  // for Scalar
#include <kaad/static_vector.hpp>           // for StaticVector
#include <kaad/tensor/tensor_types.hpp>     // for ShapeView, Shape, Strides
#include <span>                             // for span
#include <utility>                          // for make_index_sequence, ind...

namespace kaad::operations {

struct Slice {

    static constexpr std::size_t ARITY = 1;

    static constexpr const char *OPERATION_NAME = "slice";

    static Shape make_res_shape(std::array<INode *, 1> input, ShapeView shape,
                                StaticVector<std::size_t> start);

    struct ForwardParams {

        const Scalar *inp_begin;
        Scalar *res_begin;

        Strides eff_inp;
        Strides eff_res;

        StaticVector<std::size_t>
            inp_start_offset; ///< Per axis offset applied to input tensor.

        Shape res_shape;

        ForwardParams(std::array<INode *, 1> input, INode *result,
                      [[maybe_unused]] ShapeView shape,
                      std::span<const std::size_t> start);
    };

    template <std::size_t res_rank, std::size_t axis>
    static void forward_walk(const ForwardParams &params,
                             std::size_t inp_offset,
                             std::size_t res_offset) noexcept {

        for (std::size_t i = 0; i < params.res_shape[axis]; i++) {

            const std::size_t INP_IDX = params.inp_start_offset[axis] +
                                        inp_offset + (i * params.eff_inp[axis]);
            const std::size_t RES_IDX = res_offset + (i * params.eff_res[axis]);

            if constexpr (axis >= res_rank - 1) {

                params.res_begin[RES_IDX] = params.inp_begin[INP_IDX];

            } else {

                forward_walk<res_rank, axis + 1>(params, INP_IDX, RES_IDX);
            }
        }
    }

    using forward_fn = void (*)(const ForwardParams &params);

    template <std::size_t res_rank>
    static void forward(const ForwardParams &params) {

        forward_walk<res_rank, 0>(params, 0, 0);
    }

    struct BackwardParams : ForwardParams {

        Scalar *d_inp_begin;
        const Scalar *d_res_begin;

        BackwardParams(std::array<INode *, 1> input, INode *result,
                       ShapeView shape, std::span<const std::size_t> start)
            : ForwardParams(input, result, shape, start),
              d_inp_begin(input[0]->gradient_mut().data()),
              d_res_begin(result->gradient().data()) {}
    };

    template <std::size_t res_rank, std::size_t axis>
    static void backward_walk(const BackwardParams &params,
                              std::size_t d_inp_offset,
                              std::size_t d_res_offset) noexcept {

        for (std::size_t i = 0; i < params.res_shape[axis]; i++) {

            const std::size_t D_INP_IDX = d_inp_offset +
                                          params.inp_start_offset[axis] +
                                          (i * params.eff_inp[axis]);

            const std::size_t D_RES_IDX =
                d_res_offset + (i * params.eff_res[axis]);

            if constexpr (axis >= res_rank - 1) {

                params.d_inp_begin[D_INP_IDX] += params.d_res_begin[D_RES_IDX];

            } else {

                backward_walk<res_rank, axis + 1>(params, D_INP_IDX, D_RES_IDX);
            }
        }
    }

    using backward_fn = void (*)(const BackwardParams &params);

    template <std::size_t res_rank>
    static void backward(const BackwardParams &params) {

        backward_walk<res_rank, 0>(params, 0, 0);
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

    static Dispatch
    dispatch([[maybe_unused]] std::array<INode *, 1> input, INode *result,
             [[maybe_unused]] ShapeView shape,
             [[maybe_unused]] std::span<const std::size_t> start) {
        // -1 because of +1 in make table function
        return {
            .forward = make_forward_dispatch_table(
                std::make_index_sequence<KAAD_MAX_RANK>())[result->rank() - 1],
            .backward = make_backward_dispatch_table(
                std::make_index_sequence<KAAD_MAX_RANK>())[result->rank() - 1]};
    }
};

static_assert(Operation<Slice>);

} // namespace kaad::operations
