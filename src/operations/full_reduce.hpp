#pragma once

#include "../graph/operation_concept.hpp"
#include "../operations/reduction_policies.hpp"
#include <cstdint>
#include <kaad/graph/inode.hpp>

namespace kaad::operations {

template <ReductionPolicy Policy> struct FullReduce {

    static constexpr std::size_t ARITY = 1;

    static constexpr const char *OPERATION_NAME = Policy::OPERATION_NAME;

    static Shape
    make_res_shape([[maybe_unused]] std::array<INode *, 1> input) noexcept {
        return SCALAR_SHAPE;
    }

    struct ForwardParams {

        const Scalar *inp_begin;
        const Scalar *inp_end;

        Scalar *res_begin;

        Scalar inp_size; ///< Number of elements in the input tensor.

        ForwardParams(std::array<INode *, 1> input, INode *result)
            : inp_begin(input[0]->value().data()),
              inp_end(input[0]->value().data() + input[0]->value().size()),
              res_begin(result->value_mut().data()),
              inp_size(static_cast<Scalar>(input[0]->value().size())) {}
    };

    using forward_fn = void (*)(const ForwardParams &params);

    static void forward(const ForwardParams &params) noexcept {

        const Scalar *inp = params.inp_begin;

        for (; inp < params.inp_end; inp++) {
            *params.res_begin += *inp;
        }

        if constexpr (std::is_same_v<Policy, MeanPolicy>) {
            *params.res_begin /= params.inp_size;
        }
    }

    struct BackwardParams : ForwardParams {

        Scalar *d_inp_begin;
        const Scalar *d_inp_end;

        const Scalar *d_res_begin;

        BackwardParams(std::array<INode *, 1> input, INode *result)
            : ForwardParams(input, result),
              d_inp_begin(input[0]->gradient_mut().data()),
              d_inp_end(input[0]->gradient().data() +
                        input[0]->gradient().size()),
              d_res_begin(result->gradient().data()) {}
    };

    using backward_fn = void (*)(const BackwardParams &params);

    static void backward(const BackwardParams &params) noexcept {

        Scalar *d_inp = params.d_inp_begin;

        Scalar gradient_val = *params.d_res_begin;

        if constexpr (std::is_same_v<Policy, MeanPolicy>) {
            gradient_val /= params.inp_size;
        }

        for (; d_inp != params.d_inp_end; d_inp++) {
            *d_inp += gradient_val;
        }
    }

    struct Dispatch {
        forward_fn forward;
        backward_fn backward;
    };

    static Dispatch dispatch([[maybe_unused]] std::array<INode *, 1> input,
                             [[maybe_unused]] INode *result) {

        return {.forward = forward, .backward = backward};
    }
};

using FullReduceMean = FullReduce<MeanPolicy>;
static_assert(Operation<FullReduceMean>);

using FullReduceSum = FullReduce<SumPolicy>;
static_assert(Operation<FullReduceSum>);

} // namespace kaad::operations
