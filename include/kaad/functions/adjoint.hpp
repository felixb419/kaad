#pragma once

#include "primal.hpp"  // for batch_matmul, matmul
#include <concepts>    // for concept
#include <cstddef>     // for size_t
#include <type_traits> // for declval

namespace kaad {

/**
 * @namespace functions::adjoint
 * @brief Contains adjoint (e.g. used for backward computation) tensor
 * operations.
 */
namespace functions::adjoint {

/**
 * @defgroup adjoint_functions Adjoint (e.g. used for backward computation)
 * tensor operations.
 */

/**
 * @namespace kaad::tensoruncs::adjoint
 */
namespace binary {

/**
 * @defgroup binary_adjoint_functions Adjoint functions that take two inputs.
 * @ingroup adjoint_functions
 */

/**
 * @brief Concept requiring a kernel to have:
 * @ingroup binary_adjoint_functions
 * 1. 'value_type' alias
 * 2. static void Op(const value_type&, value_type&, const value_type&,
 *                   value_type&, const value_type&, const value_type&);
 */
template <class Kernel>
concept kernel_class = requires { typename Kernel::value_type; } && requires {
    {
        Kernel::Grad(std::declval<const typename Kernel::value_type &>(),
                     std::declval<typename Kernel::value_type &>(),
                     std::declval<const typename Kernel::value_type &>(),
                     std::declval<typename Kernel::value_type &>(),
                     std::declval<const typename Kernel::value_type &>(),
                     std::declval<const typename Kernel::value_type &>())
    } -> std::same_as<void>;
};

template <kernel_class Kernel> constexpr bool kernel_noexcept() {
    return noexcept(
        Kernel::Grad(std::declval<const typename Kernel::value_type &>(),
                     std::declval<typename Kernel::value_type &>(),
                     std::declval<const typename Kernel::value_type &>(),
                     std::declval<typename Kernel::value_type &>(),
                     std::declval<const typename Kernel::value_type &>(),
                     std::declval<const typename Kernel::value_type &>()));
}

template <kernel_class Kernel>
using pointwise_fn = void (*)(const typename Kernel::value_type *lhs,
                              typename Kernel::value_type *d_lhs,
                              const typename Kernel::value_type *rhs,
                              typename Kernel::value_type *d_rhs,
                              const typename Kernel::value_type *res,
                              const typename Kernel::value_type *d_res,
                              const typename Kernel::value_type *end);

template <kernel_class Kernel>
using flexible_fn = void (*)(const typename Kernel::value_type *lhs,
                             typename Kernel::value_type *d_lhs,
                             const typename Kernel::value_type *rhs,
                             typename Kernel::value_type *d_rhs,
                             const typename Kernel::value_type *res,
                             const typename Kernel::value_type *d_res,
                             int *stride_lhs, int *stride_rhs, int *stride_res,
                             std::size_t *res_dim_offset, int res_rank);

template <typename T>
using dot_fn = void (*)(const T *lhs, T *d_lhs, const T *rhs, T *d_rhs,
                        const T *d_res, const T *lhs_end);

template <typename T>
using matmul_fn = void (*)(const T *lhs, T *d_lhs, const T *rhs, T *d_rhs,
                           const T *res, const T *d_res, int *lhs_rows,
                           int *rhs_cols, int *shared_dim, int *stride_lhs,
                           int *stride_rhs, int *stride_res);

template <typename T>
using batch_matmul_fn = void (*)(const T *lhs, T *d_lhs, const T *rhs, T *d_rhs,
                                 const T *d_res, int **stride_lhs,
                                 int **stride_rhs, int **stride_res,
                                 int **res_shape, int *lhs_dim_offset,
                                 int *rhs_dim_offset, int *shared_dim,
                                 int res_rank);

/**
 * @brief Accumulates the gradient of Op, @p lhs , @p rhs .
 * @ingroup binary_adjoint_functions
 * @pre @p lhs and @p res have the same shape and @p rhs is 1-rank.
 * @pre Every operand must have the same shape as their gradient.
 * @tparam Kernel A struct containing a static binary funcion ('Grad').
 * @param[in] lhs Pointer to the start of tensor.
 * @param[out] d_lhs Pointer to the start of the gradient w.r.t. @p tensor.
 * @param[in] rhs Pointer to the start of 0-rank tensor.
 * @param[out] d_rhs Pointer to the start of the gradient w.r.t. @p rhs.
 * @param[in] res Pointer to the start of tensor (tensor).
 * @param[in] d_res Pointer to the start of the gradient w.r.t. @p res.
 * @param res_end Pointer to the end of @p res.
 */
template <kernel_class Kernel>
void scalarRhs(const typename Kernel::value_type *lhs,
               typename Kernel::value_type *d_lhs,
               const typename Kernel::value_type *rhs,
               typename Kernel::value_type *d_rhs,
               const typename Kernel::value_type *res,
               const typename Kernel::value_type *d_res,
               const typename Kernel::value_type
                   *res_end) noexcept(kernel_noexcept<Kernel>()) {
    for (; res != res_end; lhs++, d_lhs++, res++, d_res++) {
        Kernel::Grad(*lhs, *d_lhs, *rhs, *d_rhs, *res, *d_res);
    }
}

/**
 * @brief Accumulates the gradient of Op, @p lhs , @p rhs .
 * @ingroup binary_adjoint_functions
 * @pre @p rhs and @p res have the same shape and @p lhs is 0-rank.
 * @pre Every operand must have the same shape as their gradient.
 * @tparam Kernel A struct containing a static binary funcion ('Grad').
 * @param[in] lhs Pointer to the start of 0-rank tensor.
 * @param[out] d_lhs Pointer to the start of the gradient w.r.t. @p lhs.
 * @param[in] rhs Pointer to the start of tensor.
 * @param[out] d_rhs Pointer to the start of the gradient w.r.t. @p rhs.
 * @param[in] res Pointer to the start of tensor.
 * @param[in] d_res Pointer to the start of the gradient w.r.t. @p res.
 * @param res_end Pointer to the end of @p res.
 */
template <kernel_class Kernel>
void scalarLhs(const typename Kernel::value_type *lhs,
               typename Kernel::value_type *d_lhs,
               const typename Kernel::value_type *rhs,
               typename Kernel::value_type *d_rhs,
               const typename Kernel::value_type *res,
               const typename Kernel::value_type *d_res,
               const typename Kernel::value_type
                   *res_end) noexcept(kernel_noexcept<Kernel>()) {
    for (; res != res_end; rhs++, d_rhs++, res++, d_res++) {
        Kernel::Grad(*lhs, *d_lhs, *rhs, *d_rhs, *res, *d_res);
    }
}

/**
 * @brief Accumulates the gradient of Op, @p lhs , @p rhs .
 * @ingroup binary_adjoint_functions
 * @pre @p lhs, @p rhs and @p res have the same shape.
 * @pre Every operand must have the same shape as their gradient.
 * @tparam Kernel A struct containing a static binary funcion ('Grad').
 * @param[in] lhs Pointer to the start of tensor.
 * @param[out] d_lhs Pointer to the start of the gradient w.r.t. @p lhs.
 * @param[in] rhs Pointer to the start of tensor.
 * @param[out] d_rhs Pointer to the start of the gradient w.r.t. @p rhs.
 * @param[in] res Pointer to the start of tensor.
 * @param[in] d_res Pointer to the start of the gradient w.r.t. @p res.
 * @param res_end Pointer to the end of @p res.
 */
template <kernel_class Kernel>
void pointwise(const typename Kernel::value_type *lhs,
               typename Kernel::value_type *d_lhs,
               const typename Kernel::value_type *rhs,
               typename Kernel::value_type *d_rhs,
               const typename Kernel::value_type *res,
               const typename Kernel::value_type *d_res,
               const typename Kernel::value_type
                   *res_end) noexcept(kernel_noexcept<Kernel>()) {
    for (; res != res_end; lhs++, d_lhs++, rhs++, d_rhs++, res++, d_res++) {
        Kernel::Grad(*lhs, *d_lhs, *rhs, *d_rhs, *res, *d_res);
    }
}

/**
 * @brief Accumulates the gradient of Op, @p lhs , @p rhs .
 * @ingroup binary_adjoint_functions
 * @pre @p res shape is the result of broadcasting @p lhs and @p rhs.
 * @pre Every operand must have the same shape as their gradient.
 * @tparam Kernel A struct containing a static binary funcion ('Grad').
 * @param[in] lhs Pointer to the start of tensor.
 * @param[out] d_lhs Pointer to the start of the gradient w.r.t. @p lhs.
 * @param[in] rhs Pointer to the start of tensor.
 * @param[out] d_rhs Pointer to the start of the gradient w.r.t. @p rhs.
 * @param[in] res Pointer to the start of tensor.
 * @param[in] d_res Pointer to the start of the gradient w.r.t. @p res.
 * @param stride_lhs Stride array of @p lhs.
 * @param stride_rhs Stride array of @p rhs.
 * @param stride_res Stride array of @p res.
 * @param res_dim_offset Offset to the end of @p res per dimension.
 * @param res_rank Number of dimensions of @p res.
 */
template <kernel_class Kernel>
void flexible(const typename Kernel::value_type *lhs,
              typename Kernel::value_type *d_lhs,
              const typename Kernel::value_type *rhs,
              typename Kernel::value_type *d_rhs,
              const typename Kernel::value_type *res,
              const typename Kernel::value_type *d_res, int *stride_lhs,
              int *stride_rhs, int *stride_res, std::size_t *res_dim_offset,
              int res_rank) noexcept(kernel_noexcept<Kernel>()) {
    const typename Kernel::value_type *end = res + *res_dim_offset;
    if (res_rank <= 1) {
        for (; res != end; lhs += *stride_lhs, rhs += *stride_rhs,
                           res += *stride_res, d_lhs += *stride_lhs,
                           d_rhs += *stride_rhs, d_res += *stride_res) {
            Kernel::Grad(*lhs, *d_lhs, *rhs, *d_rhs, *res, *d_res);
        }
    } else {
        for (; res < end; lhs += *stride_lhs, rhs += *stride_rhs,
                          res += *stride_res, d_lhs += *stride_lhs,
                          d_rhs += *stride_rhs, d_res += *stride_res) {
            flexible<Kernel>(lhs, d_lhs, rhs, d_rhs, res, d_res, stride_lhs + 1,
                             stride_rhs + 1, stride_res + 1, res_dim_offset + 1,
                             res_rank - 1);
        }
    }
}

/**
 * @brief Compile-time recursive version of runtime @ref flexible().
 * @ingroup binary_adjoint_functions
 */
template <kernel_class Kernel, int res_rank>
void flexible(const typename Kernel::value_type *lhs,
              typename Kernel::value_type *d_lhs,
              const typename Kernel::value_type *rhs,
              typename Kernel::value_type *d_rhs,
              const typename Kernel::value_type *res,
              const typename Kernel::value_type *d_res, int *stride_lhs,
              int *stride_rhs, int *stride_res, std::size_t *res_dim_offset,
              [[maybe_unused]] int _) noexcept(kernel_noexcept<Kernel>()) {
    const typename Kernel::value_type *end = res + *res_dim_offset;
    if constexpr (res_rank <= 1) {
        for (; res != end; lhs += *stride_lhs, rhs += *stride_rhs,
                           res += *stride_res, d_lhs += *stride_lhs,
                           d_rhs += *stride_rhs, d_res += *stride_res) {
            Kernel::Grad(*lhs, *d_lhs, *rhs, *d_rhs, *res, *d_res);
        }
    } else {
        for (; res != end; lhs += *stride_lhs, rhs += *stride_rhs,
                           res += *stride_res, d_lhs += *stride_lhs,
                           d_rhs += *stride_rhs, d_res += *stride_res) {
            flexible<Kernel, res_rank - 1>(
                lhs, d_lhs, rhs, d_rhs, res, d_res, stride_lhs + 1,
                stride_rhs + 1, stride_res + 1, res_dim_offset + 1, 0);
        }
    }
}

/**
 * @brief Accumulates the gradient of the dot-product of @p lhs and @p rhs.
 * @ingroup binary_adjoint_functions
 * @pre @p lhs is a 1-rank and @p rhs and @p res is 0-rank.
 * @pre Every operand must have the same shape as their gradient.
 * @tparam T Element type.
 * @tparam Kernel (Only neede for signature)
 * @param[in] lhs Pointer to the start of 1-rank tensor.
 * @param[out] d_lhs Pointer to the start of the gradient w.r.t. @p lhs.
 * @param[in] rhs Pointer to the start of 0-rank tensor.
 * @param[out] d_rhs Pointer to the start of the gradient w.r.t. @p rhs.
 * @param[in] res Pointer to the start of 0-rank tensor.
 * @param[in] d_res Pointer to the start of the gradient w.r.t. @p res.
 * @param lhs_end Pointer to the end of @p lhs.
 */
template <typename T>
void scalarDot(const T *lhs, T *d_lhs, const T *rhs, T *d_rhs, const T *d_res,
               const T *lhs_end) noexcept {
    for (; lhs != lhs_end; lhs++, d_lhs++) {
        *d_lhs += *d_res * (*rhs);
        *d_rhs += *d_res * (*lhs);
    }
}

/**
 * @brief Accumulates the gradient of the dot-product of @p lhs and
 * @ingroup binary_adjoint_functions
 * @p rhs.
 * @pre @p lhs and @p rhs are 2-rank and @p res is 0-rank.
 * @pre Every operand must have the same shape as their gradient.
 * @tparam T Element type.
 * @tparam Kernel (Only neede for signature)
 * @param[in] lhs Pointer to the start of 1-rank tensor.
 * @param[out] d_lhs Pointer to the start of the gradient w.r.t. @p lhs.
 * @param[in] rhs Pointer to the start of 1-rank tensor.
 * @param[out] d_rhs Pointer to the start of the gradient w.r.t. @p rhs.
 * @param[in] res Pointer to the start of 0-rank tensor.
 * @param[in] d_res Pointer to the start of the gradient w.r.t. @p res.
 * @param lhs_end Pointer to the end of @p lhs.
 */
template <typename T>
void dot(const T *lhs, T *d_lhs, const T *rhs, T *d_rhs, const T *d_res,
         const T *lhs_end) noexcept {
    for (; lhs != lhs_end; lhs++, d_lhs++, rhs++, d_rhs++) {
        *d_lhs += *d_res * (*rhs);
        *d_rhs += *d_res * (*lhs);
    }
}

/**
 * @brief Accumulates the gradient of Op, @p lhs , @p rhs .
 * @ingroup binary_adjoint_functions
 * @pre @p lhs, @p rhs and @p res have compatible shapes.
 * @pre Every operand must have the same shape as their gradient.
 * @tparam T Element type.
 * @param[in] lhs Pointer to the start of 2-rank tensor.
 * @param[out] d_lhs Pointer to the start of the gradient w.r.t. @p lhs.
 * @param[in] rhs Pointer to the start of 2-rank tensor.
 * @param[out] d_rhs Pointer to the start of the gradient w.r.t. @p rhs.
 * @param[in] res Pointer to the start of 2-rank tensor.
 * @param[in] d_res Pointer to the start of the gradient w.r.t. @p res.
 * @param lhs_rows Number of rows in @p lhs.
 * @param rhs_cols Number of columns in @p rhs.
 * @param shared_dim Length of the shared dimension of @p lhs and @p rhs.
 * @param stride_lhs Stride array of @p lhs.
 * @param stride_rhs Stride array of @p rhs.
 * @param stride_res Stride array of @p res.
 */
template <typename T>
void matmul(const T *lhs, T *d_lhs, const T *rhs, T *d_rhs, const T *res,
            const T *d_res, int *lhs_rows, int *rhs_cols, int *shared_dim,
            int *stride_lhs, int *stride_rhs, int *stride_res) noexcept {
    // d_lhs = d_res * rhs^T
    functions::primal::binary::matmul(d_res, rhs, d_lhs, lhs_rows[0],
                                      rhs_cols[0], shared_dim[0], stride_res,
                                      stride_rhs, stride_lhs);
    // d_rhs = lhs^T * d_res
    functions::primal::binary::matmul(
        lhs, d_res, d_rhs, lhs_rows[1], rhs_cols[1], shared_dim[1],
        stride_lhs + 2, stride_res + 2, stride_rhs + 2);
}

/**
 * @brief Accumulates the gradient of Op, @p lhs , @p rhs .
 * @ingroup binary_adjoint_functions
 * @pre @p lhs, @p rhs and @p res have compatible shapes.
 * @pre Every operand must have the same shape as their gradient.
 * @tparam T Element type.
 * @param[in] lhs Pointer to the start of tensor.
 * @param[out] d_lhs Pointer to the start of the gradient w.r.t. @p lhs.
 * @param[in] rhs Pointer to the start of tensor.
 * @param[out] d_rhs Pointer to the start of the gradient w.r.t. @p rhs.
 * @param[in] res Pointer to the start of tensor.
 * @param[in] d_res Pointer to the start of the gradient w.r.t. @p res.
 * @param stride_lhs Pointer to stride arrays ({res.stride, lhs.stride^T}).
 * @param stride_rhs Pointer to stride arrays ({rhs.stride^T, res.stride}).
 * @param stride_res Pointer to stride arrays ({lhs.stride, rhs.stride}).
 * @param res_shape Pointer to shape arrays ({lhs.shape, rhs.shape}).
 * @param lhs_dim_offset Pointer to array of step sizes ({res, lhs^T}).
 * @param rhs_dim_offset Pointer to array of step sizes ({rhs^T, res}).
 * @param shared_dim Pointer to array of length of shared dimensions.
 * @param res_rank Number of dimensions of @p res.
 */
template <typename T>
void batch_matmul(const T *lhs, T *d_lhs, const T *rhs, T *d_rhs,
                  const T *d_res, int **stride_lhs, int **stride_rhs,
                  int **stride_res, int **res_shape, int *lhs_dim_offset,
                  int *rhs_dim_offset, int *shared_dim, int res_rank) noexcept {
    // d_lhs = d_res * rhs^T
    functions::primal::binary::batch_matmul<T>(
        d_res, rhs, d_lhs, stride_res[0], stride_rhs[0], stride_lhs[0],
        res_shape[0], lhs_dim_offset[0], rhs_dim_offset[0], shared_dim[0],
        res_rank);
    // d_rhs = lhs^T * d_res
    functions::primal::binary::batch_matmul<T>(
        lhs, d_res, d_rhs, stride_lhs[1], stride_res[1], stride_rhs[1],
        res_shape[1], lhs_dim_offset[1], rhs_dim_offset[1], shared_dim[1],
        res_rank);
}

/**
 * @brief Compile-time recursive version of runtime @ref batch_matmul().
 * @ingroup binary_adjoint_functions
 */
template <typename T, int res_rank>
void batch_matmul(const T *lhs, T *d_lhs, const T *rhs, T *d_rhs,
                  const T *d_res, int **stride_lhs, int **stride_rhs,
                  int **stride_res, int **res_shape, int *lhs_dim_offset,
                  int *rhs_dim_offset, int *shared_dim,
                  [[maybe_unused]] int _) noexcept {
    // d_lhs = d_res * rhs^T
    functions::primal::binary::batch_matmul<T, res_rank>(
        d_res, rhs, d_lhs, stride_res[0], stride_rhs[0], stride_lhs[0],
        res_shape[0], lhs_dim_offset[0], rhs_dim_offset[0], shared_dim[0], 0);
    // d_rhs = lhs^T * d_res
    functions::primal::binary::batch_matmul<T, res_rank>(
        lhs, d_res, d_rhs, stride_lhs[1], stride_res[1], stride_rhs[1],
        res_shape[1], lhs_dim_offset[1], rhs_dim_offset[1], shared_dim[1], 0);
}

} // namespace binary

/**
 * @namespace kaad::functions::adjoint::unary
 */
namespace unary {

/**
 * @defgroup unary_adjoint_functions Adjoint functions that take one input.
 * @ingroup adjoint_functions
 */

/**
 * @brief Concept requiring a kernel to have:
 * @ingroup unary_adjoint_functions
 * 1. 'value_type' alias
 * 2. static void Op(const value_type&, value_type&, const value_type&, const
 * value_type&);
 */
template <class Kernel>
concept kernel_class = requires { typename Kernel::value_type; } && requires {
    {
        Kernel::Grad(std::declval<const typename Kernel::value_type &>(),
                     std::declval<typename Kernel::value_type &>(),
                     std::declval<const typename Kernel::value_type &>(),
                     std::declval<const typename Kernel::value_type &>())
    } -> std::same_as<void>;
};

template <kernel_class Kernel> constexpr bool kernel_noexcept() {
    return noexcept(
        Kernel::Grad(std::declval<const typename Kernel::value_type &>(),
                     std::declval<typename Kernel::value_type &>(),
                     std::declval<const typename Kernel::value_type &>(),
                     std::declval<const typename Kernel::value_type &>()));
}

template <kernel_class Kernel>
using pointwise_fn = void (*)(const typename Kernel::value_type *inp,
                              typename Kernel::value_type *d_inp,
                              const typename Kernel::value_type *res,
                              const typename Kernel::value_type *d_res,
                              const typename Kernel::value_type *end);

template <typename T>
using sum_dim_fn = void (*)(T *d_inp, const T *d_res, int *stride_inp,
                            int *stride_res, std::size_t *inp_dim_offset,
                            int inp_rank);

template <typename T>
using mean_fn = void (*)(T *d_inp, const T *d_res, const T *d_inp_end,
                         T divisor);

template <typename T>
using mean_dim_fn = void (*)(const T *inp, T *d_inp, const T *res,
                             const T *d_res, int *stride_inp, int *stride_res,
                             std::size_t *inp_dim_offset, int inp_rank,
                             T divisor, const T *res_end);

template <typename T>
using slice_fn = void (*)(T *d_inp, const T *d_res, int *stride_inp,
                          int *stride_res, std::size_t *start_offset,
                          std::size_t *res_dim_offset, int rank);

/**
 * @brief Accumulates the gradient of Op, @p inp .
 * @ingroup unary_adjoint_functions
 * @pre @p res is 0-rank.
 * @pre Every operand must have the same shape as their gradient.
 * @tparam Kernel A struct containing a static binary funcion ('Grad').
 * @param[in] inp Pointer to the start of tensor.
 * @param[out] d_inp Pointer to the start of the gradient w.r.t. @p inp.
 * @param[in] res Pointer to the start of 0-rank tensor.
 * @param[in] d_res Pointer to the start of the gradient w.r.t. @p res.
 * @param inp_end Pointer to the end of @p inp.
 */
template <kernel_class Kernel>
void scalarOut(const typename Kernel::value_type *inp,
               typename Kernel::value_type *d_inp,
               const typename Kernel::value_type *res,
               const typename Kernel::value_type *d_res,
               const typename Kernel::value_type
                   *inp_end) noexcept(kernel_noexcept<Kernel>()) {
    for (; inp != inp_end; inp++, d_inp++) {
        Kernel::Grad(*inp, *d_inp, *res, *d_res);
    }
}

/**
 * @brief Accumulates the gradient of Op, @p inp .
 * @ingroup unary_adjoint_functions
 * @pre @p inp and @p res have the same shape.
 * @pre Every operand must have the same shape as their gradient.
 * @tparam Kernel A struct containing a static binary funcion ('Grad').
 * @param[in] inp Pointer to the start of tensor.
 * @param[out] d_inp Pointer to the start of the gradient w.r.t. @p inp.
 * @param[in] res Pointer to the start of tensor.
 * @param[in] d_res Pointer to the start of the gradient w.r.t. @p res.
 * @param res_end Pointer to the end of @p res.
 */
template <kernel_class Kernel>
void pointwise(const typename Kernel::value_type *inp,
               typename Kernel::value_type *d_inp,
               const typename Kernel::value_type *res,
               const typename Kernel::value_type *d_res,
               const typename Kernel::value_type
                   *res_end) noexcept(kernel_noexcept<Kernel>()) {
    for (; res != res_end; inp++, d_inp++, res++, d_res++) {
        Kernel::Grad(*inp, *d_inp, *res, *d_res);
    }
}

/**
 * @brief Accumulates the gradient of sum_dim, @p inp .
 * @ingroup unary_adjoint_functions
 * @tparam T Element type
 * @param[out] d_inp Pointer to the start of the gradient w.r.t. @p inp.
 * @param[in] d_res Pointer to the start of the gradient w.r.t. @p res.
 * @param stride_inp Stride array for @p d_inp.
 * @param stride_res Stride array for @p d_res.
 * @param inp_dim_offset Offset array per dimension
 * @param inp_rank Number of dimensions
 */
template <typename T>
void sum_dim(T *d_inp, const T *d_res, int *stride_inp, int *stride_res,
             std::size_t *inp_dim_offset, int inp_rank) noexcept {
    const T *end = d_inp + *inp_dim_offset;
    if (inp_rank <= 1) {
        for (; d_inp != end; d_inp += *stride_inp, d_res += *stride_res) {
            *d_inp += *d_res;
        }
    } else {
        for (; d_inp != end; d_inp += *stride_inp, d_res += *stride_res) {
            sum_dim(d_inp, d_res, stride_inp + 1, stride_res + 1,
                    inp_dim_offset + 1, inp_rank - 1);
        }
    }
}

/**
 * @brief Compile-time recursive version of runtime @ref sum_dim().
 * @ingroup unary_adjoint_functions
 */
template <typename T, int inp_rank>
void sum_dim(T *d_inp, const T *d_res, int *stride_inp, int *stride_res,
             std::size_t *inp_dim_offset, [[maybe_unused]] int _) noexcept {
    const T *end = d_inp + *inp_dim_offset;
    if constexpr (inp_rank <= 1) {
        for (; d_inp != end; d_inp += *stride_inp, d_res += *stride_res) {
            *d_inp += *d_res;
        }
    } else {
        for (; d_inp != end; d_inp += *stride_inp, d_res += *stride_res) {
            sum_dim<T, inp_rank - 1>(d_inp, d_res, stride_inp + 1,
                                     stride_res + 1, inp_dim_offset + 1, 0);
        }
    }
}

/**
 * @brief Accumulates the gradient of mean_dim, @p inp .
 * @ingroup unary_adjoint_functions
 * @tparam T Element type
 * @param[out] d_inp Pointer to the start of the gradient w.r.t. @p inp.
 * @param[in] d_res Pointer to the start of the gradient w.r.t. @p res.
 * @param d_inp_end Pointer to the end of @p d_inp.
 * @param divisor Length of @p d_inp.
 */
template <typename T>
void mean(T *d_inp, const T *d_res, const T *d_inp_end, T divisor) noexcept {
    T mean = *d_res / divisor;
    for (; d_inp != d_inp_end; d_inp++) {
        *d_inp += mean;
    }
}

/**
 * @brief Accumulates the gradient of mean_dim, @p inp .
 * @ingroup unary_adjoint_functions
 * @tparam T Element type
 * @param[out] d_inp Pointer to the start of the gradient w.r.t. @p inp.
 * @param[in] d_res Pointer to the start of the gradient w.r.t. @p res.
 * @param stride_inp Stride array for @p d_inp.
 * @param stride_res Stride array for @p d_res.
 * @param inp_dim_offset Offset array per dimension
 * @param inp_rank Number of dimensions
 * @param divisor Length of relevant dimension.
 * @param d_inp_end Pointer to the end of @p d_inp.
 */
template <typename T>
void mean_dim(const T *inp, T *d_inp, const T *res, const T *d_res,
              int *stride_inp, int *stride_res, std::size_t *inp_dim_offset,
              int inp_rank, T divisor, const T *d_inp_end) noexcept {
    sum_dim(d_inp, d_res, stride_inp, stride_res, inp_dim_offset, inp_rank);
    for (; d_inp != d_inp_end; d_inp++) {
        *d_inp /= divisor;
    }
}

/**
 * @brief Compile-time recursive version of runtime @ref mean_dim().
 * @ingroup unary_adjoint_functions
 */
template <typename T, int inp_rank>
void mean_dim(const T *inp, T *d_inp, const T *res, const T *d_res,
              int *stride_inp, int *stride_res, std::size_t *inp_dim_offset,
              [[maybe_unused]] int _, T divisor, const T *d_inp_end) noexcept {
    sum_dim<T, inp_rank>(d_inp, d_res, stride_inp, stride_res, inp_dim_offset,
                         0);
    for (; d_inp != d_inp_end; d_inp++) {
        *d_inp /= divisor;
    }
}

/**
 * @brief Accumulates the gradient of slice, @p inp .
 * @ingroup unary_adjoint_functions
 * @tparam T Element type
 * @param[out] d_inp Pointer to the start of the gradient w.r.t. @p inp.
 * @param[in] d_res Pointer to the start of the gradient w.r.t. @p res.
 * @param stride_inp Stride array for @p inp.
 * @param stride_res Stride array for @p res.
 * @param start_offset Offset for the start of @p res in @p inp.
 * @param res_dim_offset Offset to the end of @p res per dimension.
 * @param rank Number of dimensions in @p inp and @p res.
 */
template <typename T>
void slice(T *d_inp, const T *d_res, int *stride_inp, int *stride_res,
           std::size_t *start_offset, std::size_t *res_dim_offset,
           int rank) noexcept {
    d_inp += *start_offset;
    const T *end = d_res + *res_dim_offset;
    if (rank <= 1) {
        for (; d_res != end; d_inp += *stride_inp, d_res += *stride_res) {
            *d_inp = *d_res;
        }
    } else {
        for (; d_res < end; d_inp += *stride_inp, d_res += *stride_res) {
            slice(d_inp, d_res, stride_inp + 1, stride_res + 1,
                  start_offset + 1, res_dim_offset + 1, rank - 1);
        }
    }
}

/**
 * @brief Compile-time recursive version of runtime @ref slice().
 * @ingroup unary_adjoint_functions
 */
template <typename T, int rank>
void slice(T *d_inp, const T *d_res, int *stride_inp, int *stride_res,
           std::size_t *start_offset, std::size_t *res_dim_offset,
           [[maybe_unused]] int _) noexcept {
    d_inp += *start_offset;
    const T *end = d_res + *res_dim_offset;
    if constexpr (rank <= 1) {
        for (; d_res != end; d_inp += *stride_inp, d_res += *stride_res) {
            *d_inp = *d_res;
        }
    } else {
        for (; d_res < end; d_inp += *stride_inp, d_res += *stride_res) {
            slice<T, rank - 1>(d_inp, d_res, stride_inp + 1, stride_res + 1,
                               start_offset + 1, res_dim_offset + 1, 0);
        }
    }
}

} // namespace unary
} // namespace functions::adjoint
} // namespace kaad
