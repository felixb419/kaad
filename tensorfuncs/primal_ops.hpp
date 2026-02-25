#pragma once

#include <cstddef> // for std::size_t
#include <utility> // for std::declval

namespace kaad {

/**
 * @file tensorfuncs.hpp
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
 *   (3, 1, 5) and (1, 4, 5) → broadcast to (3, 4, 5)
 *   (2, 3) and (3,) → broadcast to (2, 3)
 *
 * @note These rules are identical to NumPy’s broadcasting semantics.
 */

/**
 * @namespace tensorfuncs::primal
 * @brief Contains primal (e.g. used for forward computation) tensor operations.
 */
namespace tensorfuncs::primal {

/**
 * @namespace kaad::tensorfuncs::adjoint::binary
 */
namespace binary {

/**
 * @brief Concept requiring a kernel to have:
 * 1. 'value_type' alias
 * 2. static void Op(const value_type&, const value_type&, value_type&);
 */
template <class Kernel>
concept kernel_class = requires { typename Kernel::value_type; } && requires {
    {
        Kernel::Op(std::declval<const typename Kernel::value_type &>(),
                   std::declval<const typename Kernel::value_type &>(),
                   std::declval<typename Kernel::value_type &>())
    } -> std::same_as<void>;
};

template <kernel_class Kernel> constexpr bool kernel_noexcept() {
    return noexcept(
        Kernel::Op(std::declval<const typename Kernel::value_type &>(),
                   std::declval<const typename Kernel::value_type &>(),
                   std::declval<typename Kernel::value_type &>()));
}

template <kernel_class Kernel>
using pointwise_fn = void (*)(const typename Kernel::value_type *lhs,
                              const typename Kernel::value_type *rhs,
                              typename Kernel::value_type *res,
                              const typename Kernel::value_type *res_end);

template <kernel_class Kernel>
using flexible_fn = void (*)(const typename Kernel::value_type *lhs,
                             const typename Kernel::value_type *rhs,
                             typename Kernel::value_type *res, int *stride_lhs,
                             int *stride_rhs, int *stride_res,
                             std::size_t *res_dim_offset, int res_rank);

template <typename T>
using dot_fn = void (*)(const T *lhs, const T *rhs, T *res, const T *lhs_end);

template <typename T>
using matmul_fn = void (*)(const T *lhs, const T *rhs, T *res, int lhs_rows,
                           int rhs_cols, int shared_dim, int *stride_lhs,
                           int *stride_rhs, int *stride_res);

template <typename T>
using batch_matmul_fn = void (*)(const T *lhs, const T *rhs, T *res,
                                 int *stride_lhs, int *stride_rhs,
                                 int *stride_res, int *res_shape,
                                 int lhs_dim_offset, int rhs_dim_offset,
                                 int shared_dim, int res_rank);

/**
 * @brief Applies Op(lhs,rhs) to lhs(tensor) and rhs(scalar).
 * @pre @p lhs and @p res have the same shape and @p rhs is scalar.
 * @tparam Kernel A struct containing a static binary function ('Op').
 * @param[in] lhs Pointer to the start of lhs(tensor).
 * @param[in] rhs Pointer to rhs(scalar).
 * @param[out] res Pointer to the start of res(tensor).
 * @param res_end Pointer to the end of @p res.
 */
template <kernel_class Kernel>
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
 * @brief Applies Op(lhs,rhs) to lhs(scalar) and rhs(tensor).
 * @pre @p rhs and @p res have the same shape and @p lhs is scalar.
 * @tparam Kernel A struct containing a static binary function ('Op').
 * @param[in] lhs Pointer to lhs(scalar).
 * @param[in] rhs Pointer to the start of rhs(tensor).
 * @param[out] res Pointer to the start of res(tensor).
 * @param res_end Pointer to the end of @p res.
 */
template <kernel_class Kernel>
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
 * @brief Applies Op(lhs,rhs) to lhs(tensor) and rhs(tensor).
 * @pre @p lhs, @p rhs and @p res have the same shape.
 * @tparam Kernel A struct containing a static binary function ('Op').
 * @param[in] lhs Pointer to the start of lhs(tensor).
 * @param[in] rhs Pointer to the start of rhs(tensor).
 * @param[out] res Pointer to the start of res(tensor).
 * @param res_end Pointer to the end of @p res.
 */
template <kernel_class Kernel>
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
 * @brief Applies Op(lhs,rhs) to lhs(tensor) and rhs(tensor).
 * @pre @p res shape is the result of broadcasting @p lhs and @p rhs.
 * @tparam Kernel A struct containing a static binary function ('Op').
 * @param[in] lhs Pointer to the start of lhs(tensor).
 * @param[in] rhs Pointer to the start of rhs(tensor).
 * @param[out] res Pointer to the start of res(tensor).
 * @param stride_lhs Stride array of lhs.
 * @param stride_rhs Stride array of rhs.
 * @param stride_res Stride array of res.
 * @param res_dim_offset Offset to the end of @p res per dimension.
 * @param res_rank Number of dimensions of res.
 */
template <kernel_class Kernel>
void flexible(const typename Kernel::value_type *lhs,
              const typename Kernel::value_type *rhs,
              typename Kernel::value_type *res, int *stride_lhs,
              int *stride_rhs, int *stride_res, std::size_t *res_dim_offset,
              int rank) noexcept(kernel_noexcept<Kernel>()) {
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
 */
template <kernel_class Kernel, int rank>
void flexible(const typename Kernel::value_type *lhs,
              const typename Kernel::value_type *rhs,
              typename Kernel::value_type *res, int *stride_lhs,
              int *stride_rhs, int *stride_res, std::size_t *res_dim_offset,
              int _) noexcept(kernel_noexcept<Kernel>()) {
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
 * @brief Computes the dot product of lhs and rhs into res.
 * @pre @p lhs is a vector and rhs and res are scalars.
 * @tparam T Element type
 * @tparam Kernel (Only needed for signature).
 * @param[in] lhs Pointer to the start of lhs(vector).
 * @param[in] rhs Pointer to rhs(scalar)
 * @param[out] res Pointer to res(scalar).
 * @param lhs_end Pointer to the end of @p lhs.
 */
template <typename T>
void scalarDot(const T *lhs, const T *rhs, T *res, const T *lhs_end) noexcept {
    for (; lhs != lhs_end; lhs++) {
        *res += *lhs * (*rhs);
    }
}

/**
 * @brief Computes the dot product of lhs and rhs into res.
 * @pre @p lhs and @p rhs are vectors and res is a scalar.
 * @tparam T Element type
 * @tparam Kernel (Only needed for signature).
 * @param[in] lhs Pointer to the start of lhs(vector).
 * @param[in] rhs Pointer to the start of rhs(vector)
 * @param[out] res Pointer to res(scalar).
 * @param lhs_end Pointer to the end of @p lhs.
 */
template <typename T>
void dot(const T *lhs, const T *rhs, T *res, const T *lhs_end) noexcept {
    for (; lhs != lhs_end; lhs++, rhs++) {
        *res += *lhs * (*rhs);
    }
}

/**
 * @brief Computes matrix product of lhs and rhs into res.
 * @pre @p lhs, @p rhs and @p res have compatible shapes.
 * @tparam T Element type
 * @param[in] lhs Pointer to the start of lhs(matrix).
 * @param[in] rhs Pointer to the start of rhs(matrix).
 * @param[out] res Pointer to the start of res(matrix).
 * @param lhs_rows Number of rows in @p lhs.
 * @param rhs_cols Number of columns in @p rhs.
 * @param stride_lhs Stride array of lhs.
 * @param stride_rhs Stride array of rhs.
 * @param stride_res Stride array of res.
 */
template <typename T>
void matmul(const T *lhs, const T *rhs, T *res, int lhs_rows, int rhs_cols,
            int shared_dim, int *stride_lhs, int *stride_rhs,
            int *stride_res) noexcept {
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
 * @brief Computes Matrix product of lhs and rhs into res (treats additional
 * dimensions as batch dimensions).
 * @pre @p lhs, @p rhs and @p res have compatible shapes.
 * @tparam T Element type
 * @param[in] lhs Pointer to the start of lhs(tensor).
 * @param[in] rhs Pointer to the start of rhs(tensor).
 * @param[out] res Pointer to the start of res(tensor).
 * @param stride_lhs Stride array of lhs.
 * @param stride_rhs Stride array of rhs.
 * @param stride_res Stride array of res.
 * @param Pointer to shape array of res.
 * @param lhs_dim_offset Step size for lhs.
 * @param rhs_dim_offset Step size for rhs.
 * @param shared_dim Shared inner dimension.
 * @param res_rank Number of dimension of res.
 */
template <typename T>
void batch_matmul(const T *lhs, const T *rhs, T *res, int *stride_lhs,
                  int *stride_rhs, int *stride_res, int *res_shape,
                  int lhs_dim_offset, int rhs_dim_offset, int shared_dim,
                  int res_rank) noexcept {
    const T *end = res + (*res_shape) * (*stride_res);
    if (res_rank <= 1) {
        for (int i = 0; i < *res_shape;
             i++, lhs += *stride_lhs, rhs += *stride_rhs, res += *stride_res) {
            const T *row_lhs = lhs, *col_rhs = rhs;
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
 */
template <typename T, int res_rank>
void batch_matmul(const T *lhs, const T *rhs, T *res, int *stride_lhs,
                  int *stride_rhs, int *stride_res, int *res_shape,
                  int lhs_dim_offset, int rhs_dim_offset, int shared_dim,
                  int _) noexcept {
    const T *end = res + (*res_shape) * (*stride_res);
    if constexpr (res_rank <= 1) {
        for (int i = 0; i < *res_shape;
             i++, lhs += *stride_lhs, rhs += *stride_rhs, res += *stride_res) {
            const T *row_lhs = lhs, *col_rhs = rhs;
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

/**
 * @brief Concept requiring a kernel to have:
 * 1. 'value_type' alias
 * 2. static void Op(const value_type&, value_type&);
 */
template <class Kernel>
concept kernel_class = requires { typename Kernel::value_type; } && requires {
    {
        Kernel::Op(std::declval<const typename Kernel::value_type &>(),
                   std::declval<typename Kernel::value_type &>())
    } -> std::same_as<void>;
};

template <kernel_class Kernel> constexpr bool kernel_noexcept() {
    return noexcept(
        Kernel::Op(std::declval<const typename Kernel::value_type &>(),
                   std::declval<typename Kernel::value_type &>()));
}

template <kernel_class Kernel>
using pointwise_fn = void (*)(const typename Kernel::value_type *lhs,
                              typename Kernel::value_type *res,
                              const typename Kernel::value_type *res_end);

template <typename T>
using sum_dim_fn = void (*)(const T *lhs, T *res, int *stride_lhs,
                            int *stride_res, std::size_t *lhs_offset,
                            int res_rank);

template <typename T>
using mean_fn = void (*)(const T *lhs, T *res, const T *lhs_end, T divisor);

template <typename T>
using mean_dim_fn = void (*)(const T *lhs, T *res, int *stride_lhs,
                             int *stride_res, std::size_t *lhs_offset,
                             int res_rank, T divisor, const T *res_end);

template <typename T>
using slice_fn = void (*)(const T *lhs, T *res, int *stride_lhs,
                          int *stride_res, std::size_t *start_offset_a,
                          std::size_t *res_dim_offset, int res_rank);

/**
 * @brief Applies a unary operation to lhs(tensor).
 * @tparam Kernel A struct containing a static unary function ('Op').
 * @param[in] lhs Pointer to the start of lhs(tensor).
 * @param[out] res Pointer to res(scalar).
 * @param lhs_end Pointer to the end of lhs.
 */
template <kernel_class Kernel>
void scalarOut(const typename Kernel::value_type *lhs,
               typename Kernel::value_type *res,
               const typename Kernel::value_type
                   *lhs_end) noexcept(kernel_noexcept<Kernel>()) {
    for (; lhs != lhs_end; lhs++) {
        Kernel::Op(*lhs, *res);
    }
}

/**
 * @brief Applies a unary operation to lhs(tensor).
 * @tparam Kernel A struct containing a static unary function ('Op').
 * @param[in] lhs Pointer to the start of lhs(tensor).
 * @param[out] res Pointer to the start of res(tensor)
 * @param lhs_end Pointer to the end of res.
 * @param op Instance of the callable class.
 */
template <kernel_class Kernel>
void pointwise(const typename Kernel::value_type *lhs,
               typename Kernel::value_type *res,
               const typename Kernel::value_type
                   *res_end) noexcept(kernel_noexcept<Kernel>()) {
    for (; res != res_end; lhs++, res++) {
        Kernel::Op(*lhs, *res);
    }
}

/**
 * @brief Sums lhs(Tensor) along a dimension into res(Tensor).
 * @pre Shape of res needs to be same as lhs with relevant dimension removed.
 * @tparam T Element type
 * @param lhs Pointer to the start of lhs(Tensor).
 * @param res Pointer to the start of res(Tensor).
 * @param stride_lhs Stride array of lhs.
 * @param strideB Stride array of B.
 * @param stride_res Stride array of res.
 * @param lhs_offset Offset to the end of @p lhs per dimension.
 * @param lhs_rank Number of dimensions of lhs.
 */
template <typename T>
void sum_dim(const T *lhs, T *res, int *stride_lhs, int *stride_res,
             std::size_t *lhs_offset, int lhs_rank) noexcept {
    const T *end = lhs + *lhs_offset;
    if (lhs_rank <= 1) {
        for (; lhs != end; lhs += *stride_lhs, res += *stride_res) {
            *res += *lhs;
        }
    } else {
        for (; lhs != end; lhs += *stride_lhs, res += *stride_res) {
            sum_dim(lhs, res, stride_lhs + 1, stride_res + 1, lhs_offset + 1,
                    lhs_rank - 1);
        }
    }
}

/**
 * @brief Compile-time recursive version of runtime @ref sum_dim().
 */
template <typename T, int lhs_rank>
void sum_dim(const T *lhs, T *res, int *stride_lhs, int *stride_res,
             std::size_t *lhs_offset, int _) noexcept {
    const T *end = lhs + *lhs_offset;
    if constexpr (lhs_rank <= 1) {
        for (; lhs != end; lhs += *stride_lhs, res += *stride_res) {
            *res += *lhs;
        }
    } else {
        for (; lhs != end; lhs += *stride_lhs, res += *stride_res) {
            sum_dim<T, lhs_rank - 1>(lhs, res, stride_lhs + 1, stride_res + 1,
                                     lhs_offset + 1, 0);
        }
    }
}

/**
 * @brief Computes the mean of all elements in lhs(tensor).
 * @tparam T Element type
 * @param[in] lhs Pointer to the start of lhs(tensor).
 * @param[out] lhs Pointer to the start of res(tensor).
 * @param lhs_end Pointer to the end of lhs.
 * @param divisor Number of elements
 */
template <typename T>
void mean(const T *lhs, T *res, const T *lhs_end, T divisor) noexcept {
    for (; lhs != lhs_end; lhs++) {
        *res += *lhs;
    }
    *res /= divisor;
}

/**
 * @brief Computes mean of lhs(tensor) along a given dimension.
 * @tparam T Element type
 * @param[in] lhs Pointer to the start of lhs(tensor).
 * @param[out] res Pointer to the start of res(tensor).
 * @param stride_lhs Stride array for lhs.
 * @param stride_res Stride array for res.
 * @param lhs_offset Offset array per dimension
 * @param lhs_rank Number of dimensions in lhs.
 * @param divisor divisor to compute mean of lhs (length of dimension summed
 * over)
 * @param res_end Pointer to the end of output tensor res.
 */
template <typename T>
void mean_dim(const T *lhs, T *res, int *stride_lhs, int *stride_res,
              std::size_t *lhs_offset, int lhs_rank, T divisor,
              const T *res_end) noexcept {
    sum_dim(lhs, res, stride_lhs, stride_res, lhs_offset, lhs_rank);
    for (; res != res_end; res++) {
        *res /= divisor;
    }
}

/**
 * @brief Compile-time recursive version of runtime @ref mean_dim().
 */
template <typename T, int N>
void mean_dim(const T *lhs, T *res, int *stride_lhs, int *stride_res,
              std::size_t *lhs_offset, int _, T divisor,
              const T *res_end) noexcept {
    sum_dim<T, N>(lhs, res, stride_lhs, stride_res, lhs_offset, 0);
    for (; res != res_end; res++) {
        *res /= divisor;
    }
}

/**
 * @brief Copies a sliced view of lhs into res based on offset and stride.
 * @tparam T Element type
 * @param[in] lhs Pointer to the start of lhs(tensor).
 * @param[out] res Pointer to the start of res(tensor).
 * @param stride_lhs Stride array for lhs.
 * @param stride_res Stride array for res.
 * @param start_offset_a Offset to apply to lhs.
 * @param res_dim_offset Size of output slice.
 * @param rank Number of dimensions in lhs and res.
 */
template <typename T>
void slice(const T *lhs, T *res, int *stride_lhs, int *stride_res,
           std::size_t *start_offset_a, std::size_t *res_dim_offset,
           int rank) noexcept {
    lhs += *start_offset_a;
    const T *end = res + *res_dim_offset;
    if (rank <= 1) {
        for (; res != end; lhs += *stride_lhs, res += *stride_res) {
            *res = *lhs;
        }
    } else {
        for (; res < end; lhs += *stride_lhs, res += *stride_res) {
            slice(lhs, res, stride_lhs + 1, stride_res + 1, start_offset_a + 1,
                  res_dim_offset + 1, rank - 1);
        }
    }
}

/**
 * @brief Compile-time recursive version of runtime @ref slice().
 */
template <typename T, int rank>
void slice(const T *lhs, T *res, int *stride_lhs, int *stride_res,
           std::size_t *start_offset_a, std::size_t *res_dim_offset,
           int _) noexcept {
    lhs += *start_offset_a;
    const T *end = res + *res_dim_offset;
    if constexpr (rank <= 1) {
        for (; res != end; lhs += *stride_lhs, res += *stride_res) {
            *res = *lhs;
        }
    } else {
        for (; res < end; lhs += *stride_lhs, res += *stride_res) {
            slice<T, rank - 1>(lhs, res, stride_lhs + 1, stride_res + 1,
                               start_offset_a + 1, res_dim_offset + 1, 0);
        }
    }
}

} // namespace unary
} // namespace tensorfuncs::primal
} // namespace kaad
