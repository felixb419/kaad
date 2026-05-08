#pragma once

#include "../graph/operation_concept.hpp" // for Operation
#include <array>                          // for array
#include <cstddef>                        // for size_t
#include <kaad/graph/inode.hpp>           // for INode
#include <kaad/scalar.hpp>                // for Scalar
#include <kaad/tensor/tensor_types.hpp>   // for Shape, Strides
#include <span>                           // for span
#include <utility>                        // for pair

namespace kaad::operations {

struct Transpose {

    static constexpr std::size_t ARITY = 1;

    static constexpr const char *OPERATION_NAME = "transpose";

    /// @note will throw ShapeError if input.rank() less than 2 or throw
    /// ArgumentError if perm is invalid.
    static std::pair<Shape, Strides>
    make_res_shape(std::array<INode *, 1> input,
                   std::span<const std::size_t> perm);

    struct ForwardParams {

        const Scalar *inp_begin;
        const Scalar *inp_end;

        Scalar *res_begin;

        ForwardParams(std::array<INode *, 1> input, INode *result,
                      std::span<const std::size_t> perm);
    };

    using forward_fn = void (*)(const ForwardParams &params);

    static void forward(const ForwardParams &params);

    struct BackwardParams {

        Scalar *d_inp_begin;
        const Scalar *d_inp_end;

        const Scalar *d_res_begin;

        BackwardParams(std::array<INode *, 1> input, INode *result,
                       [[maybe_unused]] std::span<const std::size_t> perm)
            : d_inp_begin(input[0]->gradient_mut().data()),
              d_inp_end(input[0]->gradient().data() + input[0]->value().size()),
              d_res_begin(result->gradient().data()) {}
    };

    using backward_fn = void (*)(const BackwardParams &params);

    static void backward(const BackwardParams &params);

    struct Dispatch {
        forward_fn forward;
        backward_fn backward;
    };

    static Dispatch dispatch(std::array<INode *, 1> input, INode *result,
                             std::span<const std::size_t> perm);
};

static_assert(Operation<Transpose>);

} // namespace kaad::operations
