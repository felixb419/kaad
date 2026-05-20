#pragma once

#include "../operations/operation_concept.hpp"
#include "kaad/graph/internal/inode.hpp"
#include "kaad/tensor/internal/tensor_types.hpp"

#include <array>
#include <cstddef>
#include <kaad/scalar.hpp>
#include <span>
#include <utility>

namespace kaad::operations {

struct Transpose {

    static constexpr std::size_t ARITY = 1;

    static constexpr const char *OPERATION_NAME = "transpose";

    struct Metadata {
        StaticVector<std::size_t> perm;

        Metadata(std::span<const std::size_t> perm) : perm(perm) {}
    };

    /// @throws kaad::ShapeError If input.rank() less than 2.
    /// @throws kaad::ArgumentError If perm is invalid.
    static std::pair<Shape, Strides>
    make_res_shape(std::array<INode *, 1> input, const Metadata &mdata);

    struct ForwardParams {

        const Scalar *inp_begin;
        const Scalar *inp_end;

        Scalar *res_begin;

        ForwardParams() = default;

        ForwardParams(std::array<INode *, 1> input, INode *result,
                      const Metadata &mdata);
    };

    using forward_fn = void (*)(const ForwardParams &params);

    static void forward(const ForwardParams &params);

    struct BackwardParams {

        Scalar *d_inp_begin;
        const Scalar *d_inp_end;

        const Scalar *d_res_begin;

        BackwardParams() = default;

        BackwardParams(std::array<INode *, 1> input, INode *result,
                       [[maybe_unused]] const Metadata &mdata)
            : d_inp_begin(input[0]->gradient.data),
              d_inp_end(input[0]->gradient.data + input[0]->value.size),
              d_res_begin(result->gradient.data) {}
    };

    using backward_fn = void (*)(const BackwardParams &params);

    static void backward(const BackwardParams &params);

    struct Dispatch {
        forward_fn forward;
        backward_fn backward;
    };

    static Dispatch dispatch(std::array<INode *, 1> input, INode *result,
                             const Metadata &mdata);
};

static_assert(Operation<Transpose>);

} // namespace kaad::operations
