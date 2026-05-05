#pragma once

#include <array>                                  // for array
#include <cstddef>                                // for size_t
#include <kaad/graph/inode.hpp>                   // for INode
#include <kaad/graph/operation_concept.hpp>       // for Operation
#include <kaad/max_rank.hpp>                      // for KAAD_MAX_RANK
#include <kaad/operations/reduction_policies.hpp> // for MeanPolicy, SumPolicy
#include <kaad/scalar.hpp>                        // for Scalar
#include <kaad/tensor/tensor_types.hpp>           // for Shape, Strides
#include <type_traits>                            // for is_same_v
#include <utility>                                // for index_sequence, mak...

namespace kaad::operations {

namespace internal {

Shape make_res_shape_impl(std::array<INode *, 1> input, std::size_t dim,
                          bool keep_rank);

void fwdparams_ctr_impl(const Scalar *&inp_begin, Scalar *&res_begin,
                        const Scalar *&res_end, Strides &eff_inp,
                        Strides &eff_res, Shape &inp_shape,
                        Scalar &relevant_dim_extent,
                        std::array<INode *, 1> input, INode *result,
                        std::size_t relevant_dim, bool keep_rank);

} // namespace internal

template <reduction_policy Policy> struct Reduce {

    static constexpr std::size_t ARITY = 1;

    static constexpr const char *OPERATION_NAME = Policy::OPERATION_NAME;

    /// @note Will throw ArgumentError if @p relevant_dim is not a valid index
    /// of input.shape(), and ShapeError if input.rank() is less than 2.
    static Shape make_res_shape(std::array<INode *, 1> input, std::size_t dim,
                                bool keep_rank) {
        return internal::make_res_shape_impl(input, dim, keep_rank);
    }

    struct ForwardParams {

        const Scalar *inp_begin;
        Scalar *res_begin;

        const Scalar *res_end;

        Strides eff_inp;
        Strides eff_res;

        Shape inp_shape;

        Scalar relevant_dim_extent; ///< Extent of the dimension reduced over.

        ForwardParams(std::array<INode *, 1> input, INode *result,
                      std::size_t relevant_dim, bool keep_rank) {
            internal::fwdparams_ctr_impl(
                this->inp_begin, this->res_begin, this->res_end, this->eff_inp,
                this->eff_res, this->inp_shape, this->relevant_dim_extent,
                input, result, relevant_dim, keep_rank);
        }
    };

    template <std::size_t inp_rank, std::size_t dim>
    static void forward_walk(const ForwardParams &params,
                             std::size_t inp_offset,
                             std::size_t res_offset) noexcept {

        for (std::size_t i = 0; i < params.inp_shape[dim]; i++) {

            const std::size_t RES_IDX = res_offset + (i * params.eff_res[dim]);
            const std::size_t INP_IDX = inp_offset + (i * params.eff_inp[dim]);

            if constexpr (dim >= inp_rank - 1) {

                params.res_begin[RES_IDX] += params.inp_begin[INP_IDX];

            } else {

                forward_walk<inp_rank, dim + 1>(params, INP_IDX, RES_IDX);
            }
        }
    }

    using forward_fn = void (*)(const ForwardParams &params);

    template <std::size_t inp_rank>
    static void forward(const ForwardParams &params) {

        forward_walk<inp_rank, 0>(params, 0, 0);

        // if policy is mean, divide res by extent of relevant dim
        if constexpr (std::is_same_v<Policy, MeanPolicy>) {

            for (Scalar *res = params.res_begin; res < params.res_end; res++) {
                *res /= params.relevant_dim_extent;
            }
        }
    }

    struct BackwardParams : ForwardParams {

        Scalar *d_inp_begin;
        const Scalar *d_inp_end;

        const Scalar *d_res_begin;

        BackwardParams(std::array<INode *, 1> input, INode *result,
                       std::size_t relevant_dim, bool keep_rank)
            : ForwardParams(input, result, relevant_dim, keep_rank),
              d_inp_begin(input[0]->gradient_mut().data()),
              d_inp_end(input[0]->gradient().data() +
                        input[0]->gradient().size()),
              d_res_begin(result->gradient().data()) {}
    };

    template <std::size_t inp_rank, std::size_t dim>
    static void backward_walk(const BackwardParams &params,
                              std::size_t d_inp_offset,
                              std::size_t d_res_offset) noexcept {

        for (std::size_t i = 0; i < params.inp_shape[dim]; i++) {

            const std::size_t D_RES_IDX =
                d_res_offset + (i * params.eff_res[dim]);
            const std::size_t D_INP_IDX =
                d_inp_offset + (i * params.eff_inp[dim]);

            if constexpr (dim >= inp_rank - 1) {

                params.d_inp_begin[D_INP_IDX] += params.d_res_begin[D_RES_IDX];

            } else {

                backward_walk<inp_rank, dim + 1>(params, D_INP_IDX, D_RES_IDX);
            }
        }
    }

    using backward_fn = void (*)(const BackwardParams &params);

    template <std::size_t inp_rank>
    static void backward(const BackwardParams &params) {

        backward_walk<inp_rank, 0>(params, 0, 0);

        // if policy is mean, divide res by extent of relevant dim
        if constexpr (std::is_same_v<Policy, MeanPolicy>) {

            for (Scalar *d_inp = params.d_inp_begin; d_inp < params.d_inp_end;
                 d_inp++) {

                *d_inp /= params.relevant_dim_extent;
            }
        }
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

    static Dispatch dispatch(std::array<INode *, 1> input,
                             [[maybe_unused]] INode *result,
                             [[maybe_unused]] std::size_t relevant_dim,
                             [[maybe_unused]] bool keep_rank) {
        // -1 because of +1 in make table function
        return {
            .forward = make_forward_dispatch_table(
                std::make_index_sequence<KAAD_MAX_RANK>())[input[0]->rank() -
                                                           1],
            .backward = make_backward_dispatch_table(
                std::make_index_sequence<KAAD_MAX_RANK>())[input[0]->rank() -
                                                           1]};
    }
};

using ReduceMean = Reduce<MeanPolicy>;
static_assert(Operation<ReduceMean>);

using ReduceSum = Reduce<SumPolicy>;
static_assert(Operation<ReduceSum>);

} // namespace kaad::operations
