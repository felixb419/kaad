#pragma once

#include "../graph/operator_node.hpp" // for Operation
#include <cstdint>
#include <kaad/enums.hpp>                      // for ScalarOrder
#include <kaad/operators/internal/kernels.hpp> // for BinaryKernel, bin_kerne...

namespace kaad::operations {

struct Pointwise {
    template <kernels::Binary Kernel> struct Binary {

        static constexpr std::size_t ARITY = 2;

        static constexpr const char *OPERATION_NAME = Kernel::OPERATION_NAME;

        /// @note Throws ShapeError if shapes differ; scalar inputs are
        /// broadcast automatically.
        static Shape make_res_shape(std::array<INode *, 2> inputs) {

            ShapeView lhs_shape = inputs[0]->value().shape;
            ShapeView rhs_shape = inputs[1]->value().shape;

            StridesView lhs_strides = inputs[0]->value().strides;
            StridesView rhs_strides = inputs[1]->value().strides;

            bool shapes_match = std::ranges::equal(lhs_shape, rhs_shape) &&
                                std::ranges::equal(lhs_strides, rhs_strides);

            bool lhs_scalar = lhs_shape.empty() ||
                              (lhs_shape.size() == 1 && lhs_shape[0] == 1);
            bool rhs_scalar = rhs_shape.empty() ||
                              (lhs_shape.size() == 1 && lhs_shape[0] == 1);

            if (lhs_scalar) {
                return rhs_shape;
            }

            if (rhs_scalar) {
                return lhs_shape;
            }

            if (!shapes_match) {

                throw ShapeError("incompatible tensor shapes for " +
                                 std::string(OPERATION_NAME) + ", lhs.shape()" +
                                 to_string(lhs_shape) + ", rhs.shape()" +
                                 to_string(rhs_shape));
            }

            return lhs_shape;
        }

        struct ForwardParams {

            const Scalar *lhs_begin;
            const Scalar *rhs_begin;
            Scalar *res_begin;

            const Scalar *res_end;

            ForwardParams(std::array<INode *, 2> inputs, INode *result)
                : lhs_begin(inputs[0]->value().data()),
                  rhs_begin(inputs[1]->value().data()),
                  res_begin(result->value_mut().data()),
                  res_end(result->value().data() + result->value().size()) {}
        };

        using forward_fn = void (*)(const ForwardParams &params);

        template <ScalarOrder S>
            requires(S == NONE_SCALAR || S == LHS_IS_SCALAR ||
                     S == RHS_IS_SCALAR)
        static void forward(const ForwardParams &params) noexcept(
            kernels::binary_noexcept<Kernel>()) {

            const Scalar *lhs = params.lhs_begin;
            const Scalar *rhs = params.rhs_begin;
            Scalar *res = params.res_begin;

            if constexpr (S == NONE_SCALAR) {

                for (; res != params.res_end; lhs++, rhs++, res++) {

                    Kernel::op(*lhs, *rhs, *res);
                }

            } else if constexpr (S == LHS_IS_SCALAR) {

                for (; res != params.res_end; rhs++, res++) {

                    Kernel::op(*lhs, *rhs, *res);
                }

            } else if constexpr (S == RHS_IS_SCALAR) {

                for (; res != params.res_end; lhs++, res++) {

                    Kernel::op(*lhs, *rhs, *res);
                }
            }
        }

        struct BackwardParams : ForwardParams {

            Scalar *d_lhs_begin;
            Scalar *d_rhs_begin;
            const Scalar *d_res_begin;

            BackwardParams() = default;

            BackwardParams(std::array<INode *, 2> inputs, INode *result)
                : ForwardParams(inputs, result),
                  d_lhs_begin(inputs[0]->gradient_mut().data()),
                  d_rhs_begin(inputs[1]->gradient_mut().data()),
                  d_res_begin(result->gradient().data()) {};
        };

        using backward_fn = void (*)(const BackwardParams &params);

        template <ScalarOrder S>
            requires(S == NONE_SCALAR || S == LHS_IS_SCALAR ||
                     S == RHS_IS_SCALAR)
        static void backward(const BackwardParams &params) noexcept(
            kernels::binary_noexcept<Kernel>()) {

            const Scalar *lhs = params.lhs_begin;
            Scalar *d_lhs = params.d_lhs_begin;
            const Scalar *rhs = params.rhs_begin;
            Scalar *d_rhs = params.d_rhs_begin;
            const Scalar *res = params.res_begin;
            const Scalar *d_res = params.d_res_begin;

            if constexpr (S == NONE_SCALAR) {

                for (; res != params.res_end;
                     lhs++, d_lhs++, rhs++, d_rhs++, res++, d_res++) {

                    Kernel::grad(*lhs, *d_lhs, *rhs, *d_rhs, *res, *d_res);
                }

            } else if constexpr (S == LHS_IS_SCALAR) {

                for (; res != params.res_end; rhs++, d_rhs++, res++, d_res++) {

                    Kernel::grad(*lhs, *d_lhs, *rhs, *d_rhs, *res, *d_res);
                }

            } else if constexpr (S == RHS_IS_SCALAR) {

                for (; res != params.res_end; lhs++, d_lhs++, res++, d_res++) {

                    Kernel::grad(*lhs, *d_lhs, *rhs, *d_rhs, *res, *d_res);
                }
            }
        }

        struct Dispatch {
            forward_fn forward;
            backward_fn backward;
        };

        static Dispatch dispatch(std::array<INode *, 2> inputs,
                                 [[maybe_unused]] INode *result) {

            bool lhs_scalar = inputs[0]->value().scalar();
            bool rhs_scalar = inputs[1]->value().scalar();

            if (lhs_scalar) {
                return {.forward = forward<LHS_IS_SCALAR>,
                        .backward = backward<LHS_IS_SCALAR>};
            }

            if (rhs_scalar) {
                return {.forward = forward<RHS_IS_SCALAR>,
                        .backward = backward<RHS_IS_SCALAR>};
            }

            return {.forward = forward<NONE_SCALAR>,
                    .backward = backward<NONE_SCALAR>};
        }
    };

    static_assert(Operation<Binary<kernels::Add<Scalar>>>);

    template <kernels::Unary Kernel> struct Unary {

        static constexpr std::size_t ARITY = 1;

        static constexpr const char *OPERATION_NAME = "uinary pointwise";

        static Shape make_res_shape(std::array<INode *, 1> inputs) {
            return inputs[0]->value().shape;
        }

        struct ForwardParams {

            const Scalar *inp_begin;
            Scalar *res_begin;

            const Scalar *res_end;

            ForwardParams(std::array<INode *, 1> inputs, INode *result)
                : inp_begin(inputs[0]->value().data()),
                  res_begin(result->value_mut().data()),
                  res_end(result->value().data() + result->value().size()) {}
        };

        using forward_fn = void (*)(const ForwardParams &params);

        void static forward(const ForwardParams &params) noexcept(
            kernels::unary_noexcept<Kernel>()) {

            const Scalar *inp = params.inp_begin;
            Scalar *res = params.res_begin;

            for (; res != params.res_end; inp++, res++) {

                Kernel::op(*inp, *res);
            }
        }

        struct BackwardParams {

            const Scalar *inp_begin;
            Scalar *d_inp_begin;

            const Scalar *res_begin;
            const Scalar *d_res_begin;

            const Scalar *res_end;

            BackwardParams(std::array<INode *, 1> inputs, INode *result)
                : inp_begin(inputs[0]->value().data()),
                  d_inp_begin(inputs[0]->gradient_mut().data()),
                  res_begin(result->value().data()),
                  d_res_begin(result->gradient().data()),
                  res_end(result->value().data() + result->value().size()) {}
        };

        using backward_fn = void (*)(const BackwardParams &params);

        static void backward(const BackwardParams &params) noexcept(
            kernels::unary_noexcept<Kernel>()) {

            const Scalar *inp = params.inp_begin;
            Scalar *d_inp = params.d_inp_begin;
            const Scalar *res = params.res_begin;
            const Scalar *d_res = params.d_res_begin;

            for (; res != params.res_end; inp++, d_inp++, res++, d_res++) {

                Kernel::grad(*inp, *d_inp, *res, *d_res);
            }
        }

        struct Dispatch {
            forward_fn forward;
            backward_fn backward;
        };

        static Dispatch dispatch([[maybe_unused]] std::array<INode *, 1> inputs,
                                 [[maybe_unused]] INode *result) {
            return {.forward = forward, .backward = backward};
        }
    };

    static_assert(Operation<Unary<kernels::Log<Scalar>>>);
};

} // namespace kaad::operations
