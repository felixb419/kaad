#pragma once

#include <kaad/scalar.hpp>              // for Scalar
#include <kaad/tensor/tensor.hpp>       // for TensorViewConst
#include <kaad/tensor/tensor_types.hpp> // for Stride, ShapeView, Shape

namespace kaad::functions {

struct Matmul {

    /*
     * @brief Broadcasts @p lhs_shape and @p rhs_shape according to matrix
     * multiplication.
     * @note @p new_shape will be resized.
     * @return true if the broadcast was sucessful, false if the shapes are
     * incompatible
     */
    static bool broadcast(ShapeView lhs, ShapeView rhs, Shape &new_shape);

    struct Metadata {

        int lhs_rows;   ///< Number of rows in lhs.
        int rhs_cols;   ///< Number of cols in rhs.
        int shared_dim; ///< Size of the shared dimension of @p lhs and @p rhs.

        Strides eff_lhs; ///< Broadcasted strides for lhs.
        Strides eff_rhs; ///< Broadcasted strides for rhs.
        Strides eff_res; ///< Broadcasted strides for res.

        Metadata() = default;

        Metadata(TensorViewConst lhs, TensorViewConst rhs, TensorViewConst res);
    };

    using primal_fn = void (*)(const Scalar *lhs, const Scalar *rhs,
                               Scalar *res, const Metadata &mdata);

    /**
     * @brief Computes matrix product of @p lhs and @p rhs into @p res.
     * @ingroup binary_primal_functions
     * @pre @p lhs, @p rhs and @p res have compatible shapes.
     * @param[in] lhs Pointer to the start of rank-2 tensor.
     * @param[in] rhs Pointer to the start of rank-2 tensor.
     * @param[out] res Pointer to the start of rank-2 tensor.
     * @param mdata Metadata needed for traversal.
     */
    static void primal(const Scalar *lhs, const Scalar *rhs, Scalar *res,
                       const Metadata &mdata) noexcept;

    using adjoint_fn = void (*)(const Scalar *lhs, Scalar *d_lhs,
                                const Scalar *rhs, Scalar *d_rhs,
                                const Scalar *d_res, const Metadata &wrt_lhs,
                                const Metadata &wrt_rhs);

    /**
     * @brief Accumulates the gradient of Op, @p lhs , @p rhs .
     * @ingroup binary_adjoint_functions
     * @pre @p lhs, @p rhs and @p res have compatible shapes.
     * @pre Every operand must have the same shape as their gradient.
     * @param lhs Pointer to value elements.
     * @param d_lhs Pointer to gradient elements.
     * @param rhs Pointer to value elements.
     * @param d_rhs Pointer to gradient elements.
     * @param res Pointer to value elements.
     * @param d_res Pointer to gradient elements.
     * @param wrt_lhs Metadata for gradient w.r.t to @p lhs.
     * @param wrt_rhs Metadata for gradient w.r.t to @p rhs.
     */
    static void adjoint(const Scalar *lhs, Scalar *d_lhs, const Scalar *rhs,
                        Scalar *d_rhs, const Scalar *d_res,
                        const Metadata &wrt_lhs,
                        const Metadata &wrt_rhs) noexcept;
};

} // namespace kaad::functions
