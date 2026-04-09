#pragma once

#include <cstddef>                   // for size_t
#include <kaad/functions/primal.hpp> // for batch_matmul, matmul

/**
 * @defgroup adjoint_functions Adjoint (e.g. used for backward computation)
 * tensor operations.
 */

/**
 * @defgroup binary_adjoint_functions Adjoint functions that take two inputs.
 * @ingroup adjoint_functions
 */

/**
 * @defgroup unary_adjoint_functions Adjoint functions that take one input.
 * @ingroup adjoint_functions
 */

/**
 * @namespace functions::adjoint
 * @brief Contains adjoint (e.g. used for backward computation) tensor
 * operations.
 */
namespace kaad::functions::adjoint {

/**
 * @namespace kaad::tensoruncs::adjoint
 * @ingroup binary_adjoint_functions
 */
namespace binary {} // namespace binary

/**
 * @namespace kaad::functions::adjoint::unary
 * @ingroup unary_adjoint_functions
 */
namespace unary {

template <unary_kernel_class Kernel>
using pointwise_fn = void (*)(const typename Kernel::value_type *inp,
                              typename Kernel::value_type *d_inp,
                              const typename Kernel::value_type *res,
                              const typename Kernel::value_type *d_res,
                              const typename Kernel::value_type *end);

template <typename T>
using sum_dim_fn = void (*)(T *d_inp, const T *d_res, stride *strides_inp,
                            stride *strides_res, std::size_t *inp_dim_offset,
                            std::size_t inp_rank);

template <typename T>
using mean_fn = void (*)(T *d_inp, const T *d_res, const T *d_inp_end,
                         T divisor);

template <typename T>
using mean_dim_fn = void (*)(T *d_inp, const T *d_res, stride *strides_inp,
                             stride *strides_res, std::size_t *inp_dim_offset,
                             std::size_t inp_rank, T divisor, const T *res_end);

template <typename T>
using slice_fn = void (*)(T *d_inp, const T *d_res, stride *strides_inp,
                          stride *strides_res, std::size_t *start_offset,
                          std::size_t *res_dim_offset, std::size_t rank);

/**
 * @brief Accumulates the gradient of Op in @p inp .
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
template <unary_kernel_class Kernel>
void scalar_out(const typename Kernel::value_type *inp,
                typename Kernel::value_type *d_inp,
                const typename Kernel::value_type *res,
                const typename Kernel::value_type *d_res,
                const typename Kernel::value_type
                    *inp_end) noexcept(un_kernel_noexcept<Kernel>()) {
    for (; inp != inp_end; inp++, d_inp++) {
        Kernel::grad(*inp, *d_inp, *res, *d_res);
    }
}

/**
 * @brief Accumulates the gradient of Op in @p inp .
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
template <unary_kernel_class Kernel>
void pointwise(const typename Kernel::value_type *inp,
               typename Kernel::value_type *d_inp,
               const typename Kernel::value_type *res,
               const typename Kernel::value_type *d_res,
               const typename Kernel::value_type
                   *res_end) noexcept(un_kernel_noexcept<Kernel>()) {
    for (; res != res_end; inp++, d_inp++, res++, d_res++) {
        Kernel::grad(*inp, *d_inp, *res, *d_res);
    }
}

/**
 * @brief Accumulates the gradient of sum_dim in @p inp .
 * @ingroup unary_adjoint_functions
 * @tparam T Element type
 * @param[out] d_inp Pointer to the start of the gradient w.r.t. @p inp.
 * @param[in] d_res Pointer to the start of the gradient w.r.t. @p res.
 * @param strides_inp Stride array for @p d_inp.
 * @param strides_res Stride array for @p d_res.
 * @param inp_dim_offset Offset array per dimension
 * @param inp_rank Number of dimensions
 */
template <typename T>
void sum_dim(T *d_inp, const T *d_res, stride *strides_inp, stride *strides_res,
             std::size_t *inp_dim_offset, std::size_t inp_rank) noexcept {
    const T *end = d_inp + *inp_dim_offset;
    if (inp_rank <= 1) {
        for (; d_inp != end; d_inp += *strides_inp, d_res += *strides_res) {
            *d_inp += *d_res;
        }
    } else {
        for (; d_inp != end; d_inp += *strides_inp, d_res += *strides_res) {
            sum_dim(d_inp, d_res, strides_inp + 1, strides_res + 1,
                    inp_dim_offset + 1, inp_rank - 1);
        }
    }
}

/**
 * @brief Compile-time recursive version of runtime @ref sum_dim().
 * @ingroup unary_adjoint_functions
 */
template <typename T, std::size_t inp_rank>
void sum_dim(T *d_inp, const T *d_res, stride *strides_inp, stride *strides_res,
             std::size_t *inp_dim_offset,
             [[maybe_unused]] std::size_t unused) noexcept {
    const T *end = d_inp + *inp_dim_offset;
    if constexpr (inp_rank <= 1) {
        for (; d_inp != end; d_inp += *strides_inp, d_res += *strides_res) {
            *d_inp += *d_res;
        }
    } else {
        for (; d_inp != end; d_inp += *strides_inp, d_res += *strides_res) {
            sum_dim<T, inp_rank - 1>(d_inp, d_res, strides_inp + 1,
                                     strides_res + 1, inp_dim_offset + 1, 0);
        }
    }
}

/**
 * @brief Accumulates the gradient of mean in @p inp .
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
 * @brief Accumulates the gradient of mean_dim in @p inp .
 * @ingroup unary_adjoint_functions
 * @tparam T Element type
 * @param[out] d_inp Pointer to the start of the gradient w.r.t. @p inp.
 * @param[in] d_res Pointer to the start of the gradient w.r.t. @p res.
 * @param strides_inp Stride array for @p d_inp.
 * @param strides_res Stride array for @p d_res.
 * @param inp_dim_offset Offset array per dimension
 * @param inp_rank Number of dimensions
 * @param divisor Length of relevant dimension.
 * @param d_inp_end Pointer to the end of @p d_inp.
 */
template <typename T>
void mean_dim(T *d_inp, const T *d_res, stride *strides_inp,
              stride *strides_res, std::size_t *inp_dim_offset,
              std::size_t inp_rank, T divisor, const T *d_inp_end) noexcept {
    sum_dim(d_inp, d_res, strides_inp, strides_res, inp_dim_offset, inp_rank);
    for (; d_inp != d_inp_end; d_inp++) {
        *d_inp /= divisor;
    }
}

/**
 * @brief Compile-time recursive version of runtime @ref mean_dim().
 * @ingroup unary_adjoint_functions
 */
template <typename T, std::size_t inp_rank>
void mean_dim(T *d_inp, const T *d_res, stride *strides_inp,
              stride *strides_res, std::size_t *inp_dim_offset,
              [[maybe_unused]] std::size_t unused, T divisor,
              const T *d_inp_end) noexcept {
    sum_dim<T, inp_rank>(d_inp, d_res, strides_inp, strides_res, inp_dim_offset,
                         0);
    for (; d_inp != d_inp_end; d_inp++) {
        *d_inp /= divisor;
    }
}

/**
 * @brief Accumulates the gradient of slice in @p inp .
 * @ingroup unary_adjoint_functions
 * @tparam T Element type
 * @param[out] d_inp Pointer to the start of the gradient w.r.t. @p inp.
 * @param[in] d_res Pointer to the start of the gradient w.r.t. @p res.
 * @param strides_inp Stride array for @p inp.
 * @param strides_res Stride array for @p res.
 * @param start_offset Offset for the start of @p res in @p inp.
 * @param res_dim_offset Offset to the end of @p res per dimension.
 * @param rank Number of dimensions in @p inp and @p res.
 */
template <typename T>
void slice(T *d_inp, const T *d_res, stride *strides_inp, stride *strides_res,
           std::size_t *start_offset, std::size_t *res_dim_offset,
           std::size_t rank) noexcept {
    d_inp += *start_offset;
    const T *end = d_res + *res_dim_offset;
    if (rank <= 1) {
        for (; d_res != end; d_inp += *strides_inp, d_res += *strides_res) {
            *d_inp = *d_res;
        }
    } else {
        for (; d_res < end; d_inp += *strides_inp, d_res += *strides_res) {
            slice(d_inp, d_res, strides_inp + 1, strides_res + 1,
                  start_offset + 1, res_dim_offset + 1, rank - 1);
        }
    }
}

/**
 * @brief Compile-time recursive version of runtime @ref slice().
 * @ingroup unary_adjoint_functions
 */
template <typename T, std::size_t rank>
void slice(T *d_inp, const T *d_res, stride *strides_inp, stride *strides_res,
           std::size_t *start_offset, std::size_t *res_dim_offset,
           [[maybe_unused]] std::size_t unused) noexcept {
    d_inp += *start_offset;
    const T *end = d_res + *res_dim_offset;
    if constexpr (rank <= 1) {
        for (; d_res != end; d_inp += *strides_inp, d_res += *strides_res) {
            *d_inp = *d_res;
        }
    } else {
        for (; d_res < end; d_inp += *strides_inp, d_res += *strides_res) {
            slice<T, rank - 1>(d_inp, d_res, strides_inp + 1, strides_res + 1,
                               start_offset + 1, res_dim_offset + 1, 0);
        }
    }
}

} // namespace unary
} // namespace kaad::functions::adjoint
