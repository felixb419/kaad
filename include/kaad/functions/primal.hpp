#pragma once

#include <cstddef>                    // for size_t
#include <kaad/functions/kernels.hpp> // for binary_kernel_class, unary_kernel_class

/**
 * @file functions.hpp
 * @brief Tensor operations with NumPy-style broadcasting.
 *
 * Broadcasting rules:
 * - When performing elementwise operations, input tensors are aligned from the
 * trailing dimensions.
 * - For each dimension:
 *   1. If the sizes match, that dimension is compatible.
 *   2. If one tensor has size 1, it is broadcast to match the other.
 *   3. If sizes differ and neither is 1, the shapes are incompatible.
 * - The resulting tensor shape is the elementwise maximum along each dimension.
 *
 * Examples:
 *   (3, 1, 5) and (1, 4, 5) -> broadcast to (3, 4, 5)
 *   (2, 3) and (3,) -> broadcast to (2, 3)
 *
 * @note These rules are identical to NumPy's broadcasting semantics.
 */

/**
 * @namespace functions::primal
 * @brief Contains primal (e.g. used for forward computation) tensor operations.
 */
namespace kaad::functions::primal {

/**
 * @defgroup primal_functions Primal (e.g. used for forward computation) tensor
 * operations.
 */

/**
 * @namespace kaad::functions::adjoint::binary
 */
namespace binary {

/**
 * @defgroup binary_primal_functions Primal functions that take two inputs.
 * @ingroup primal_functions
 */

template <binary_kernel_class Kernel> constexpr bool kernel_noexcept() {
    return noexcept(
        Kernel::Op(std::declval<const typename Kernel::value_type &>(),
                   std::declval<const typename Kernel::value_type &>(),
                   std::declval<typename Kernel::value_type &>()));
}

template <binary_kernel_class Kernel>
using pointwise_fn = void (*)(const typename Kernel::value_type *lhs,
                              const typename Kernel::value_type *rhs,
                              typename Kernel::value_type *res,
                              const typename Kernel::value_type *res_end);

template <binary_kernel_class Kernel>
using flexible_fn = void (*)(const typename Kernel::value_type *lhs,
                             const typename Kernel::value_type *rhs,
                             typename Kernel::value_type *res, int *stride_lhs,
                             int *stride_rhs, int *stride_res,
                             std::size_t *res_dim_offset, std::size_t res_rank);

template <typename T>
using dot_fn = void (*)(const T *lhs, const T *rhs, T *res, const T *lhs_end);

template <typename T>
using matmul_fn = void (*)(const T *lhs, const T *rhs, T *res, int lhs_rows,
                           int rhs_cols, int shared_dim, const int *stride_lhs,
                           const int *stride_rhs, const int *stride_res);

template <typename T>
using batch_matmul_fn = void (*)(const T *lhs, const T *rhs, T *res,
                                 const int *stride_lhs, const int *stride_rhs,
                                 const int *stride_res, const int *res_shape,
                                 int lhs_dim_offset, int rhs_dim_offset,
                                 int shared_dim, std::size_t res_rank);

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
void scalarRhs(const typename Kernel::value_type *lhs,
               const typename Kernel::value_type *rhs,
               typename Kernel::value_type *res,
               const typename Kernel::value_type
                   *res_end) noexcept(kernel_noexcept<Kernel>()) {
    for (; res != res_end; lhs++, res++) {
        Kernel::Op(*lhs, *rhs, *res);
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
void scalarLhs(const typename Kernel::value_type *lhs,
               const typename Kernel::value_type *rhs,
               typename Kernel::value_type *res,
               const typename Kernel::value_type
                   *res_end) noexcept(kernel_noexcept<Kernel>()) {
    for (; res != res_end; rhs++, res++) {
        Kernel::Op(*lhs, *rhs, *res);
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
                   *res_end) noexcept(kernel_noexcept<Kernel>()) {
    for (; res != res_end; lhs++, rhs++, res++) {
        Kernel::Op(*lhs, *rhs, *res);
    }
}

/**
 * @brief Applies Op to @p lhssand @p rhs .
 * @ingroup binary_primal_functions
 * @pre @p res shape is the result of broadcasting @p lhs and @p rhs.
 * @tparam Kernel A struct containing a static binary function ('Op').
 * @param[in] lhs Pointer to the start of tensor.
 * @param[in] rhs Pointer to the start of tensor.
 * @param[out] res Pointer to the start of tensor.
 * @param stride_lhs Stride array of @p lhs.
 * @param stride_rhs Stride array of @p rhs.
 * @param stride_res Stride array of @p res.
 * @param res_dim_offset Offset to the end of @p res per dimension.
 * @param res_rank Number of dimensions of @p res.
 */
template <binary_kernel_class Kernel>
void flexible(const typename Kernel::value_type *lhs,
              const typename Kernel::value_type *rhs,
              typename Kernel::value_type *res, int *stride_lhs,
              int *stride_rhs, int *stride_res, std::size_t *res_dim_offset,
              std::size_t rank) noexcept(kernel_noexcept<Kernel>()) {
    const typename Kernel::value_type *end = res + *res_dim_offset;
    if (rank <= 1) {
        for (; res != end;
             lhs += *stride_lhs, rhs += *stride_rhs, res += *stride_res) {
            Kernel::Op(*lhs, *rhs, *res);
        }
    } else {
        for (; res < end;
             lhs += *stride_lhs, rhs += *stride_rhs, res += *stride_res) {
            flexible<Kernel>(lhs, rhs, res, stride_lhs + 1, stride_rhs + 1,
                             stride_res + 1, res_dim_offset + 1, rank - 1);
        }
    }
}

/**
 * @brief Compile-time recursive version of the runtime @ref flexible().
 * @ingroup binary_primal_functions
 */
template <binary_kernel_class Kernel, std::size_t rank>
void flexible(
    const typename Kernel::value_type *lhs,
    const typename Kernel::value_type *rhs, typename Kernel::value_type *res,
    int *stride_lhs, int *stride_rhs, int *stride_res,
    std::size_t *res_dim_offset,
    [[maybe_unused]] std::size_t unused) noexcept(kernel_noexcept<Kernel>()) {
    const typename Kernel::value_type *end = res + *res_dim_offset;
    if constexpr (rank <= 1) {
        for (; res != end;
             lhs += *stride_lhs, rhs += *stride_rhs, res += *stride_res) {
            Kernel::Op(*lhs, *rhs, *res);
        }
    } else {
        for (; res < end;
             lhs += *stride_lhs, rhs += *stride_rhs, res += *stride_res) {
            flexible<Kernel, rank - 1>(lhs, rhs, res, stride_lhs + 1,
                                       stride_rhs + 1, stride_res + 1,
                                       res_dim_offset + 1, 0);
        }
    }
}

/**
 * @brief Computes the dot product of @p lhs and @p rhs into @p res.
 * @ingroup binary_primal_functions
 * @pre @p lhs is rank-2 and @p rhs and @p res are rank-0.
 * @tparam T Element type
 * @tparam Kernel (Only needed for signature).
 * @param[in] lhs Pointer to the start of rank-1 tensor.
 * @param[in] rhs Pointer to rank-0 tensor
 * @param[out] res Pointer to rank-0 tensor.
 * @param lhs_end Pointer to the end of @p lhs.
 */
template <typename T>
void scalarDot(const T *lhs, const T *rhs, T *res, const T *lhs_end) noexcept {
    for (; lhs != lhs_end; lhs++) {
        *res += *lhs * (*rhs);
    }
}

/**
 * @brief Computes the dot product of @p lhs and @p rhs into @p res.
 * @ingroup binary_primal_functions
 * @pre @p lhs and @p rhs are rank-1 and @p res is rank-0.
 * @tparam T Element type
 * @tparam Kernel (Only needed for signature).
 * @param[in] lhs Pointer to the start of rank-1 tensor.
 * @param[in] rhs Pointer to the start of rank-1 tensor
 * @param[out] res Pointer to rank-0 tensor.
 * @param lhs_end Pointer to the end of @p lhs.
 */
template <typename T>
void dot(const T *lhs, const T *rhs, T *res, const T *lhs_end) noexcept {
    for (; lhs != lhs_end; lhs++, rhs++) {
        *res += *lhs * (*rhs);
    }
}

/**
 * @brief Computes matrix product of @p lhs and @p rhs into @p res.
 * @ingroup binary_primal_functions
 * @pre @p lhs, @p rhs and @p res have compatible shapes.
 * @tparam T Element type
 * @param[in] lhs Pointer to the start of rank-2 tensor.
 * @param[in] rhs Pointer to the start of rank-2 tensor.
 * @param[out] res Pointer to the start of rank-2 tensor.
 * @param lhs_rows Number of rows in @p lhs.
 * @param rhs_cols Number of columns in @p rhs.
 * @param stride_lhs Stride array of @p lhs.
 * @param stride_rhs Stride array of @p rhs.
 * @param stride_res Stride array of @p res.
 */
template <typename T>
void matmul(const T *lhs, const T *rhs, T *res, int lhs_rows, int rhs_cols,
            int shared_dim, const int *stride_lhs, const int *stride_rhs,
            const int *stride_res) noexcept {
    const T *row_lhs;
    const T *col_rhs;
    const T *elem_rhs;
    for (int lhs_idx = 0; lhs_idx < lhs_rows;
         lhs_idx++, lhs += stride_lhs[0], res += stride_res[0]) {
        col_rhs = rhs;
        for (int rhs_idx = 0; rhs_idx < rhs_cols;
             rhs_idx++, col_rhs += stride_rhs[1], res += stride_res[1]) {
            row_lhs = lhs;
            elem_rhs = col_rhs;
            for (int i = 0; i < shared_dim;
                 i++, row_lhs += stride_lhs[1], elem_rhs += stride_rhs[0]) {
                *res += (*row_lhs) * (*elem_rhs);
            }
        }
    }
}

/**
 * @brief Computes Matrix product of @p lhs and @p rhs into @p res (treats
 * @ingroup binary_primal_functions
 * additional dimensions as batch dimensions).
 * @pre @p lhs, @p rhs and @p res have compatible shapes.
 * @tparam T Element type
 * @param[in] lhs Pointer to the start of tensor.
 * @param[in] rhs Pointer to the start of tensor.
 * @param[out] res Pointer to the start of tensor.
 * @param stride_lhs Stride array of @p lhs.
 * @param stride_rhs Stride array of @p rhs.
 * @param stride_res Stride array of @p res.
 * @param Pointer to shape array of @p res.
 * @param lhs_dim_offset Step size for @p lhs.
 * @param rhs_dim_offset Step size for @p rhs.
 * @param shared_dim Shared inner dimension.
 * @param res_rank Number of dimension of @p res.
 */
template <typename T>
void batch_matmul(const T *lhs, const T *rhs, T *res, const int *stride_lhs,
                  const int *stride_rhs, const int *stride_res,
                  const int *res_shape, int lhs_dim_offset, int rhs_dim_offset,
                  int shared_dim, std::size_t res_rank) noexcept {
    if (res_rank <= 1) {
        for (int i = 0; i < *res_shape;
             i++, lhs += *stride_lhs, rhs += *stride_rhs, res += *stride_res) {
            const T *row_lhs = lhs;
            const T *col_rhs = rhs;
            for (int j = 0; j < shared_dim;
                 j++, row_lhs += lhs_dim_offset, col_rhs += rhs_dim_offset) {
                *res += (*row_lhs) * (*col_rhs);
            }
        }
    } else {
        for (int i = 0; i < *res_shape;
             i++, lhs += *stride_lhs, rhs += *stride_rhs, res += *stride_res) {
            batch_matmul(lhs, rhs, res, stride_lhs + 1, stride_rhs + 1,
                         stride_res + 1, res_shape + 1, lhs_dim_offset,
                         rhs_dim_offset, shared_dim, res_rank - 1);
        }
    }
}

/**
 * @brief Compile-time recursive version of runtime @ref batch_matmul().
 * @ingroup binary_primal_functions
 */
template <typename T, std::size_t res_rank>
void batch_matmul(const T *lhs, const T *rhs, T *res, const int *stride_lhs,
                  const int *stride_rhs, const int *stride_res,
                  const int *res_shape, int lhs_dim_offset, int rhs_dim_offset,
                  int shared_dim,
                  [[maybe_unused]] std::size_t unused) noexcept {
    if constexpr (res_rank <= 1) {
        for (int i = 0; i < *res_shape;
             i++, lhs += *stride_lhs, rhs += *stride_rhs, res += *stride_res) {
            const T *row_lhs = lhs;
            const T *col_rhs = rhs;
            for (int j = 0; j < shared_dim;
                 j++, row_lhs += lhs_dim_offset, col_rhs += rhs_dim_offset) {
                *res += (*row_lhs) * (*col_rhs);
            }
        }
    } else {
        for (int i = 0; i < *res_shape;
             i++, lhs += *stride_lhs, rhs += *stride_rhs, res += *stride_res) {
            batch_matmul<T, res_rank - 1>(
                lhs, rhs, res, stride_lhs + 1, stride_rhs + 1, stride_res + 1,
                res_shape + 1, lhs_dim_offset, rhs_dim_offset, shared_dim, 0);
        }
    }
}

} // namespace binary

/**
 * @namespace kaad::Operations::unary
 */
namespace unary {

template <unary_kernel_class Kernel> constexpr bool kernel_noexcept() {
    return noexcept(
        Kernel::Op(std::declval<const typename Kernel::value_type &>(),
                   std::declval<typename Kernel::value_type &>()));
}

template <unary_kernel_class Kernel>
using pointwise_fn = void (*)(const typename Kernel::value_type *inp,
                              typename Kernel::value_type *res,
                              const typename Kernel::value_type *res_end);

template <typename T>
using sum_dim_fn = void (*)(const T *inp, T *res, int *stride_inp,
                            int *stride_res, std::size_t *inp_offset,
                            std::size_t res_rank);

template <typename T>
using mean_fn = void (*)(const T *inp, T *res, const T *inp_end, T divisor);

template <typename T>
using mean_dim_fn = void (*)(const T *inp, T *res, int *stride_inp,
                             int *stride_res, std::size_t *inp_offset,
                             std::size_t res_rank, T divisor, const T *res_end);

template <typename T>
using slice_fn = void (*)(const T *inp, T *res, int *stride_inp,
                          int *stride_res, std::size_t *start_offset_a,
                          std::size_t *res_dim_offset, std::size_t res_rank);

/**
 * @defgroup unary_primal_functions Primal functions that take one input.
 * @ingroup primal_functions
 */

/**
 * @brief Applies a unary operation to @p inp .
 * @ingroup unary_primal_functions
 * @tparam Kernel A struct containing a static unary function ('Op').
 * @param[in] inp Pointer to the start of tensor.
 * @param[out] res Pointer to rank-0 tensor.
 * @param inp_end Pointer to the end of @p inp.
 */
template <unary_kernel_class Kernel>
void scalarOut(const typename Kernel::value_type *inp,
               typename Kernel::value_type *res,
               const typename Kernel::value_type
                   *inp_end) noexcept(kernel_noexcept<Kernel>()) {
    for (; inp != inp_end; inp++) {
        Kernel::Op(*inp, *res);
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
                   *res_end) noexcept(kernel_noexcept<Kernel>()) {
    for (; res != res_end; inp++, res++) {
        Kernel::Op(*inp, *res);
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
 * @param stride_inp Stride array of @p inp.
 * @param strideB Stride array of B.
 * @param stride_res Stride array of @p res.
 * @param inp_offset Offset to the end of @p inp per dimension.
 * @param inp_rank Number of dimensions of @p inp.
 */
template <typename T>
void sum_dim(const T *inp, T *res, int *stride_inp, int *stride_res,
             std::size_t *inp_offset, std::size_t inp_rank) noexcept {
    const T *end = inp + *inp_offset;
    if (inp_rank <= 1) {
        for (; inp != end; inp += *stride_inp, res += *stride_res) {
            *res += *inp;
        }
    } else {
        for (; inp != end; inp += *stride_inp, res += *stride_res) {
            sum_dim(inp, res, stride_inp + 1, stride_res + 1, inp_offset + 1,
                    inp_rank - 1);
        }
    }
}

/**
 * @brief Compile-time recursive version of runtime @ref sum_dim().
 * @ingroup unary_primal_functions
 */
template <typename T, std::size_t inp_rank>
void sum_dim(const T *inp, T *res, int *stride_inp, int *stride_res,
             std::size_t *inp_offset,
             [[maybe_unused]] std::size_t unused) noexcept {
    const T *end = inp + *inp_offset;
    if constexpr (inp_rank <= 1) {
        for (; inp != end; inp += *stride_inp, res += *stride_res) {
            *res += *inp;
        }
    } else {
        for (; inp != end; inp += *stride_inp, res += *stride_res) {
            sum_dim<T, inp_rank - 1>(inp, res, stride_inp + 1, stride_res + 1,
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
 * @param stride_inp Stride array for @p inp.
 * @param stride_res Stride array for @p res.
 * @param inp_offset Offset array per dimension
 * @param inp_rank Number of dimensions in @p inp.
 * @param divisor divisor to compute mean of @p inp (length of dimension summed
 * over)
 * @param res_end Pointer to the end of output tensor @p res.
 */
template <typename T>
void mean_dim(const T *inp, T *res, int *stride_inp, int *stride_res,
              std::size_t *inp_offset, std::size_t inp_rank, T divisor,
              const T *res_end) noexcept {
    sum_dim(inp, res, stride_inp, stride_res, inp_offset, inp_rank);
    for (; res != res_end; res++) {
        *res /= divisor;
    }
}

/**
 * @brief Compile-time recursive version of runtime @ref mean_dim().
 * @ingroup unary_primal_functions
 */
template <typename T, std::size_t inp_rank>
void mean_dim(const T *inp, T *res, int *stride_inp, int *stride_res,
              std::size_t *inp_offset, [[maybe_unused]] std::size_t unused,
              T divisor, const T *res_end) noexcept {
    sum_dim<T, inp_rank>(inp, res, stride_inp, stride_res, inp_offset, 0);
    for (; res != res_end; res++) {
        *res /= divisor;
    }
}

/**
 * @brief Copies a sliced view of @p inp into @p res based on offset and stride.
 * @ingroup unary_primal_functions
 * @tparam T Element type
 * @param[in] inp Pointer to the start of tensor.
 * @param[out] res Pointer to the start of tensor.
 * @param stride_inp Stride array for @p inp.
 * @param stride_res Stride array for @p res.
 * @param start_offset_a Offset to apply to @p inp.
 * @param res_dim_offset Size of output slice.
 * @param rank Number of dimensions in @p inp and @p res.
 */
template <typename T>
void slice(const T *inp, T *res, int *stride_inp, int *stride_res,
           std::size_t *start_offset_a, std::size_t *res_dim_offset,
           std::size_t rank) noexcept {
    inp += *start_offset_a;
    const T *end = res + *res_dim_offset;
    if (rank <= 1) {
        for (; res != end; inp += *stride_inp, res += *stride_res) {
            *res = *inp;
        }
    } else {
        for (; res < end; inp += *stride_inp, res += *stride_res) {
            slice(inp, res, stride_inp + 1, stride_res + 1, start_offset_a + 1,
                  res_dim_offset + 1, rank - 1);
        }
    }
}

/**
 * @brief Compile-time recursive version of runtime @ref slice().
 * @ingroup unary_primal_functions
 */
template <typename T, std::size_t rank>
void slice(const T *inp, T *res, int *stride_inp, int *stride_res,
           std::size_t *start_offset_a, std::size_t *res_dim_offset,
           [[maybe_unused]] std::size_t unused) noexcept {
    inp += *start_offset_a;
    const T *end = res + *res_dim_offset;
    if constexpr (rank <= 1) {
        for (; res != end; inp += *stride_inp, res += *stride_res) {
            *res = *inp;
        }
    } else {
        for (; res < end; inp += *stride_inp, res += *stride_res) {
            slice<T, rank - 1>(inp, res, stride_inp + 1, stride_res + 1,
                               start_offset_a + 1, res_dim_offset + 1, 0);
        }
    }
}

} // namespace unary
} // namespace kaad::functions::primal
