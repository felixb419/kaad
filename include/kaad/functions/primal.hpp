#pragma once

#include <cstddef>                    // for size_t
#include <kaad/functions/kernels.hpp> // for binary_kernel_class, unary_kernel_class
#include <kaad/tensor/tensor_types.hpp> // for stride

/**
 * @defgroup primal_functions Primal (e.g. used for forward computation) tensor
 * operations.
 */

/**
 * @defgroup binary_primal_functions Primal functions that take two inputs.
 * @ingroup primal_functions
 */

/**
 * @defgroup unary_primal_functions Primal functions that take one input.
 * @ingroup primal_functions
 */

/**
 * @namespace functions::primal
 * @brief Contains primal (e.g. used for forward computation) tensor operations.
 * @ingroup primal_functions
 */
namespace kaad::functions::primal {

/**
 * @namespace kaad::functions::adjoint::binary
 * @ingroup binary_primal_functions
 */
namespace binary {

template <binary_kernel_class Kernel>
using pointwise_fn = void (*)(const typename Kernel::value_type *lhs,
                              const typename Kernel::value_type *rhs,
                              typename Kernel::value_type *res,
                              const typename Kernel::value_type *res_end);

/**
 * @brief Applies Op to @p lhssand @p rhs .
 * @ingroup binary_primal_functions
 * @pre @p lhs and @p res have the same shape and @p rhs is rank-0.
 * @tparam Kernel A struct containing a static binary function ('Op').
 * @param[in] lhs Pointer to the start of tensor.
 * @param[in] rhs Pointer to rank-0 tensor.
 * @param[out] res Pointer to the start of tensor.
 * @param res_end Pointer to the end of @p res.
 */
template <binary_kernel_class Kernel>
void scalar_rhs(const typename Kernel::value_type *lhs,
                const typename Kernel::value_type *rhs,
                typename Kernel::value_type *res,
                const typename Kernel::value_type
                    *res_end) noexcept(bin_kernel_noexcept<Kernel>()) {
    for (; res != res_end; lhs++, res++) {
        Kernel::op(*lhs, *rhs, *res);
    }
}

/**
 * @brief Applies Op to @p lhssand @p rhs .
 * @ingroup binary_primal_functions
 * @pre @p rhs and @p res have the same shape and @p lhs is rank-0.
 * @tparam Kernel A struct containing a static binary function ('Op').
 * @param[in] lhs Pointer to rank-0 tensor.
 * @param[in] rhs Pointer to the start of tensor.
 * @param[out] res Pointer to the start of tensor.
 * @param res_end Pointer to the end of @p res.
 */
template <binary_kernel_class Kernel>
void scalar_lhs(const typename Kernel::value_type *lhs,
                const typename Kernel::value_type *rhs,
                typename Kernel::value_type *res,
                const typename Kernel::value_type
                    *res_end) noexcept(bin_kernel_noexcept<Kernel>()) {
    for (; res != res_end; rhs++, res++) {
        Kernel::op(*lhs, *rhs, *res);
    }
}

/**
 * @brief Applies Op to @p lhssand @p rhs .
 * @ingroup binary_primal_functions
 * @pre @p lhs, @p rhs and @p res have the same shape.
 * @tparam Kernel A struct containing a static binary function ('Op').
 * @param[in] lhs Pointer to the start of tensor.
 * @param[in] rhs Pointer to the start of tensor.
 * @param[out] res Pointer to the start of tensor.
 * @param res_end Pointer to the end of @p res.
 */
template <binary_kernel_class Kernel>
void pointwise(const typename Kernel::value_type *lhs,
               const typename Kernel::value_type *rhs,
               typename Kernel::value_type *res,
               const typename Kernel::value_type
                   *res_end) noexcept(bin_kernel_noexcept<Kernel>()) {
    for (; res != res_end; lhs++, rhs++, res++) {
        Kernel::op(*lhs, *rhs, *res);
    }
}

} // namespace binary

/**
 * @namespace kaad::Operations::unary
 * @ingroup unary_primal_functions
 */
namespace unary {

template <unary_kernel_class Kernel>
using pointwise_fn = void (*)(const typename Kernel::value_type *inp,
                              typename Kernel::value_type *res,
                              const typename Kernel::value_type *res_end);

template <typename T>
using sum_dim_fn = void (*)(const T *inp, T *res, stride *strides_inp,
                            stride *strides_res, std::size_t *inp_offset,
                            std::size_t res_rank);

template <typename T>
using mean_fn = void (*)(const T *inp, T *res, const T *inp_end, T divisor);

template <typename T>
using mean_dim_fn = void (*)(const T *inp, T *res, stride *strides_inp,
                             stride *strides_res, std::size_t *inp_offset,
                             std::size_t res_rank, T divisor, const T *res_end);

template <typename T>
using slice_fn = void (*)(const T *inp, T *res, stride *strides_inp,
                          stride *strides_res, std::size_t *start_offset_a,
                          std::size_t *res_dim_offset, std::size_t res_rank);

/**
 * @brief Applies a unary operation to @p inp .
 * @ingroup unary_primal_functions
 * @tparam Kernel A struct containing a static unary function ('Op').
 * @param[in] inp Pointer to the start of tensor.
 * @param[out] res Pointer to rank-0 tensor.
 * @param inp_end Pointer to the end of @p inp.
 */
template <unary_kernel_class Kernel>
void scalar_out(const typename Kernel::value_type *inp,
                typename Kernel::value_type *res,
                const typename Kernel::value_type
                    *inp_end) noexcept(un_kernel_noexcept<Kernel>()) {
    for (; inp != inp_end; inp++) {
        Kernel::op(*inp, *res);
    }
}

/**
 * @brief Applies a unary operation to @p inp .
 * @ingroup unary_primal_functions
 * @tparam Kernel A struct containing a static unary function ('Op').
 * @param[in] inp Pointer to the start of tensor.
 * @param[out] res Pointer to the start of tensor
 * @param res_end Pointer to the end of @p res.
 * @param op Instance of the callable class.
 */
template <unary_kernel_class Kernel>
void pointwise(const typename Kernel::value_type *inp,
               typename Kernel::value_type *res,
               const typename Kernel::value_type
                   *res_end) noexcept(un_kernel_noexcept<Kernel>()) {
    for (; res != res_end; inp++, res++) {
        Kernel::op(*inp, *res);
    }
}

/**
 * @brief Sums @p inpsalong a dimension into @p res .
 * @ingroup unary_primal_functions
 * @pre Shape of @p res needs to be same as @p inp with relevant dimension
 * removed.
 * @tparam T Element type
 * @param inp Pointer to the start of tensor.
 * @param res Pointer to the start of tensor.
 * @param strides_inp Stride array of @p inp.
 * @param stridesB Stride array of B.
 * @param strides_res Stride array of @p res.
 * @param inp_offset Offset to the end of @p inp per dimension.
 * @param inp_rank Number of dimensions of @p inp.
 */
template <typename T>
void sum_dim(const T *inp, T *res, stride *strides_inp, stride *strides_res,
             std::size_t *inp_offset, std::size_t inp_rank) noexcept {
    const T *end = inp + *inp_offset;
    if (inp_rank <= 1) {
        for (; inp != end; inp += *strides_inp, res += *strides_res) {
            *res += *inp;
        }
    } else {
        for (; inp != end; inp += *strides_inp, res += *strides_res) {
            sum_dim(inp, res, strides_inp + 1, strides_res + 1, inp_offset + 1,
                    inp_rank - 1);
        }
    }
}

/**
 * @brief Compile-time recursive version of runtime @ref sum_dim().
 * @ingroup unary_primal_functions
 */
template <typename T, std::size_t inp_rank>
void sum_dim(const T *inp, T *res, stride *strides_inp, stride *strides_res,
             std::size_t *inp_offset,
             [[maybe_unused]] std::size_t unused) noexcept {
    const T *end = inp + *inp_offset;
    if constexpr (inp_rank <= 1) {
        for (; inp != end; inp += *strides_inp, res += *strides_res) {
            *res += *inp;
        }
    } else {
        for (; inp != end; inp += *strides_inp, res += *strides_res) {
            sum_dim<T, inp_rank - 1>(inp, res, strides_inp + 1, strides_res + 1,
                                     inp_offset + 1, 0);
        }
    }
}

/**
 * @brief Computes the mean of all elements in @p inp .
 * @ingroup unary_primal_functions
 * @tparam T Element type
 * @param[in] inp Pointer to the start of tensor.
 * @param[out] res Pointer to the start of tensor.
 * @param inp_end Pointer to the end of @p inp.
 * @param divisor Number of elements
 */
template <typename T>
void mean(const T *inp, T *res, const T *inp_end, T divisor) noexcept {
    for (; inp != inp_end; inp++) {
        *res += *inp;
    }
    *res /= divisor;
}

/**
 * @brief Computes mean of @p inpsalong a given dimension.
 * @ingroup unary_primal_functions
 * @tparam T Element type
 * @param[in] inp Pointer to the start of tensor.
 * @param[out] res Pointer to the start of tensor.
 * @param strides_inp Stride array for @p inp.
 * @param strides_res Stride array for @p res.
 * @param inp_offset Offset array per dimension
 * @param inp_rank Number of dimensions in @p inp.
 * @param divisor divisor to compute mean of @p inp (length of dimension summed
 * over)
 * @param res_end Pointer to the end of output tensor @p res.
 */
template <typename T>
void mean_dim(const T *inp, T *res, stride *strides_inp, stride *strides_res,
              std::size_t *inp_offset, std::size_t inp_rank, T divisor,
              const T *res_end) noexcept {
    sum_dim(inp, res, strides_inp, strides_res, inp_offset, inp_rank);
    for (; res != res_end; res++) {
        *res /= divisor;
    }
}

/**
 * @brief Compile-time recursive version of runtime @ref mean_dim().
 * @ingroup unary_primal_functions
 */
template <typename T, std::size_t inp_rank>
void mean_dim(const T *inp, T *res, stride *strides_inp, stride *strides_res,
              std::size_t *inp_offset, [[maybe_unused]] std::size_t unused,
              T divisor, const T *res_end) noexcept {
    sum_dim<T, inp_rank>(inp, res, strides_inp, strides_res, inp_offset, 0);
    for (; res != res_end; res++) {
        *res /= divisor;
    }
}

/**
 * @brief Copies a sliced view of @p inp into @p res based on offset and
 * strides.
 * @ingroup unary_primal_functions
 * @tparam T Element type
 * @param[in] inp Pointer to the start of tensor.
 * @param[out] res Pointer to the start of tensor.
 * @param strides_inp Stride array for @p inp.
 * @param strides_res Stride array for @p res.
 * @param start_offset_a Offset to apply to @p inp.
 * @param res_dim_offset Size of output slice.
 * @param rank Number of dimensions in @p inp and @p res.
 */
template <typename T>
void slice(const T *inp, T *res, stride *strides_inp, stride *strides_res,
           std::size_t *start_offset_a, std::size_t *res_dim_offset,
           std::size_t rank) noexcept {
    inp += *start_offset_a;
    const T *end = res + *res_dim_offset;
    if (rank <= 1) {
        for (; res != end; inp += *strides_inp, res += *strides_res) {
            *res = *inp;
        }
    } else {
        for (; res < end; inp += *strides_inp, res += *strides_res) {
            slice(inp, res, strides_inp + 1, strides_res + 1,
                  start_offset_a + 1, res_dim_offset + 1, rank - 1);
        }
    }
}

/**
 * @brief Compile-time recursive version of runtime @ref slice().
 * @ingroup unary_primal_functions
 */
template <typename T, std::size_t rank>
void slice(const T *inp, T *res, stride *strides_inp, stride *strides_res,
           std::size_t *start_offset_a, std::size_t *res_dim_offset,
           [[maybe_unused]] std::size_t unused) noexcept {
    inp += *start_offset_a;
    const T *end = res + *res_dim_offset;
    if constexpr (rank <= 1) {
        for (; res != end; inp += *strides_inp, res += *strides_res) {
            *res = *inp;
        }
    } else {
        for (; res < end; inp += *strides_inp, res += *strides_res) {
            slice<T, rank - 1>(inp, res, strides_inp + 1, strides_res + 1,
                               start_offset_a + 1, res_dim_offset + 1, 0);
        }
    }
}

} // namespace unary
} // namespace kaad::functions::primal
