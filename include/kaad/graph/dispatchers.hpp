#pragma once

#include <array>                      // for array
#include <bits/utility.h>             // for index_sequence, make_index_s...
#include <cstddef>                    // for size_t
#include <kaad/functions/adjoint.hpp> // for batch_matmul_fn, flexible_fn
#include <kaad/functions/primal.hpp>  // for batch_matmul_fn, flexible_fn
#include <kaad/max_rank.hpp>          // for KAAD_MAX_RANK

/**
 * @namespace kaad::Dispatchers
 * @brief Provides compile-time dispatcher utilities for flexible binary
 * operations and gradients.
 *
 * This namespace defines templated functions that generate arrays of function
 * pointers used to dispatch the correct binary operation or gradient
 * implementation based on tensor dimensionality.
 *
 * These utilities support dynamic shape handling by generating function arrays
 * for all valid dimensions up to `KAAD_MAX_RANK`. This enables efficient and
 * modular implementations of flexible tensor operations and their gradients.
 *
 * Typical users do not need to call these functions directly unless
 * implementing new operators.
 */
namespace kaad::Dispatchers {

// NOLINTBEGIN(readability-named-parameter)
// NOLINTBEGIN(readability-identifier-naming) // will be removed soon anyway

/// @brief Returns full table of sum_dim operation implementations.
template <typename T, std::size_t... Is>
constexpr std::array<operations::primal::unary::sum_dim_fn<T>, sizeof...(Is)>
get_sumDim_impl(std::index_sequence<Is...>) {
    return {&operations::primal::unary::sum_dim<T, Is>...};
}

template <typename T>
constexpr std::array<operations::primal::unary::sum_dim_fn<T>, KAAD_MAX_RANK>
get_sumDim() {
    return get_sumDim_impl<T>(std::make_index_sequence<KAAD_MAX_RANK>());
}

/// @brief Returns full table of sum_dim gradient implementations.
template <typename T, std::size_t... Is>
constexpr std::array<operations::adjoint::unary::sum_dim_fn<T>, sizeof...(Is)>
get_sumDim_grad_impl(std::index_sequence<Is...>) {
    return {&operations::adjoint::unary::sum_dim<T, Is>...};
}

template <typename T>
constexpr std::array<operations::adjoint::unary::sum_dim_fn<T>, KAAD_MAX_RANK>
get_sumDim_grad() {
    return get_sumDim_grad_impl<T>(std::make_index_sequence<KAAD_MAX_RANK>());
}

/// @brief Returns full table of mean_dim operation implementations.
template <typename T, std::size_t... Is>
constexpr std::array<operations::primal::unary::mean_dim_fn<T>, sizeof...(Is)>
get_meanDim_impl(std::index_sequence<Is...>) {
    return {&operations::primal::unary::mean_dim<T, Is>...};
}

template <typename T>
constexpr std::array<operations::primal::unary::mean_dim_fn<T>, KAAD_MAX_RANK>
get_meanDim() {
    return get_meanDim_impl<T>(std::make_index_sequence<KAAD_MAX_RANK>());
}

/// @brief Returns full table of mean_dim gradient implementations.
template <typename T, std::size_t... Is>
constexpr std::array<operations::adjoint::unary::mean_dim_fn<T>, sizeof...(Is)>
get_meanDim_grad_impl(std::index_sequence<Is...>) {
    return {&operations::adjoint::unary::mean_dim<T, Is>...};
}

template <typename T>
constexpr std::array<operations::adjoint::unary::mean_dim_fn<T>, KAAD_MAX_RANK>
get_meanDim_grad() {
    return get_meanDim_grad_impl<T>(std::make_index_sequence<KAAD_MAX_RANK>());
}

/// @brief Returns full table of slice operation implementations.
template <typename T, std::size_t... Is>
constexpr std::array<operations::primal::unary::slice_fn<T>, sizeof...(Is)>
get_slice_impl(std::index_sequence<Is...>) {
    return {&operations::primal::unary::slice<T, Is>...};
}

template <typename T>
constexpr std::array<operations::primal::unary::slice_fn<T>, KAAD_MAX_RANK>
get_slice() {
    return get_slice_impl<T>(std::make_index_sequence<KAAD_MAX_RANK>());
}

/// @brief Returns full table of slice gradient implementations.
template <typename T, std::size_t... Is>
constexpr std::array<operations::adjoint::unary::slice_fn<T>, sizeof...(Is)>
get_slice_grad_impl(std::index_sequence<Is...>) {
    return {&operations::adjoint::unary::slice<T, Is>...};
}

template <typename T>
constexpr std::array<operations::adjoint::unary::slice_fn<T>, KAAD_MAX_RANK>
get_slice_grad() {
    return get_slice_grad_impl<T>(std::make_index_sequence<KAAD_MAX_RANK>());
}

// NOLINTEND(readability-identifier-naming)
// NOLINTEND(readability-named-parameter)

} // namespace kaad::Dispatchers
