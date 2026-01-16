#pragma once

#include "../../tensorfuncs/adjoint_ops.hpp" // for tensorfuncs::adjoint
#include "../../tensorfuncs/primal_ops.hpp"  // for tensorfuncs::primal
#include <array>                             // for std::array
#include <cstddef>                           // for size_t
#include <utility> // for std::index_sequence, std::make_index_sequence

namespace kaad {

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
 * for all valid dimensions up to `MAX_NDIMS`. This enables efficient and
 * modular implementations of flexible tensor operations and their gradients.
 *
 * Typical users do not need to call these functions directly unless
 * implementing new operators.
 */
namespace Dispatchers {

constexpr static int MAX_NDIMS = 10;

/// @brief Returns full table of flexible binary operation implementations.
template <typename T, class Op, size_t... Is>
constexpr std::array<tensorfuncs::primal::binary::flexible_fn<T, Op>,
                     sizeof...(Is)>
get_flexOp_impl(std::index_sequence<Is...>) {
    return {&tensorfuncs::primal::binary::flexible<T, Op, Is>...};
}

template <typename T, class Op>
constexpr std::array<tensorfuncs::primal::binary::flexible_fn<T, Op>, MAX_NDIMS>
get_flexOp() {
    return get_flexOp_impl<T, Op>(std::make_index_sequence<MAX_NDIMS>());
}

/// @brief Returns full table of flexible binary gradient implementations.
template <typename T, class Grad, size_t... Is>
constexpr std::array<tensorfuncs::adjoint::binary::flexible_fn<T, Grad>,
                     sizeof...(Is)>
get_flexGrad_impl(std::index_sequence<Is...>) {
    return {&tensorfuncs::adjoint::binary::flexible<T, Grad, Is>...};
}

template <typename T, class Grad>
constexpr std::array<tensorfuncs::adjoint::binary::flexible_fn<T, Grad>,
                     MAX_NDIMS>
get_flexGrad() {
    return get_flexGrad_impl<T, Grad>(std::make_index_sequence<MAX_NDIMS>());
}

/// @brief Returns full table of batch matmul operation implementations.
template <typename T, size_t... Is>
constexpr std::array<tensorfuncs::primal::binary::batch_matmul_fn<T>,
                     sizeof...(Is)>
get_batch_matmul_impl(std::index_sequence<Is...>) {
    return {&tensorfuncs::primal::binary::batch_matmul<T, Is>...};
}

template <typename T>
constexpr std::array<tensorfuncs::primal::binary::batch_matmul_fn<T>, MAX_NDIMS>
get_batch_matmul() {
    return get_batch_matmul_impl<T>(std::make_index_sequence<MAX_NDIMS>());
}

/// @brief Returns full table of batch matmul gradient implementations.
template <typename T, size_t... Is>
constexpr std::array<tensorfuncs::adjoint::binary::batch_matmul_fn<T>,
                     sizeof...(Is)>
get_batch_matmul_grad_impl(std::index_sequence<Is...>) {
    return {&tensorfuncs::adjoint::binary::batch_matmul<T, Is>...};
}

template <typename T>
constexpr std::array<tensorfuncs::adjoint::binary::batch_matmul_fn<T>,
                     MAX_NDIMS>
get_batch_matmul_grad() {
    return get_batch_matmul_grad_impl<T>(std::make_index_sequence<MAX_NDIMS>());
}

/// @brief Returns full table of sum_dim operation implementations.
template <typename T, size_t... Is>
constexpr std::array<tensorfuncs::primal::unary::sum_dim_fn<T>, sizeof...(Is)>
get_sumDim_impl(std::index_sequence<Is...>) {
    return {&tensorfuncs::primal::unary::sum_dim<T, Is>...};
}

template <typename T>
constexpr std::array<tensorfuncs::primal::unary::sum_dim_fn<T>, MAX_NDIMS>
get_sumDim() {
    return get_sumDim_impl<T>(std::make_index_sequence<MAX_NDIMS>());
}

/// @brief Returns full table of sum_dim gradient implementations.
template <typename T, size_t... Is>
constexpr std::array<tensorfuncs::adjoint::unary::sum_dim_fn<T>, sizeof...(Is)>
get_sumDim_grad_impl(std::index_sequence<Is...>) {
    return {&tensorfuncs::adjoint::unary::sum_dim<T, Is>...};
}

template <typename T>
constexpr std::array<tensorfuncs::adjoint::unary::sum_dim_fn<T>, MAX_NDIMS>
get_sumDim_grad() {
    return get_sumDim_grad_impl<T>(std::make_index_sequence<MAX_NDIMS>());
}

/// @brief Returns full table of mean_dim operation implementations.
template <typename T, size_t... Is>
constexpr std::array<tensorfuncs::primal::unary::mean_dim_fn<T>, sizeof...(Is)>
get_meanDim_impl(std::index_sequence<Is...>) {
    return {&tensorfuncs::primal::unary::mean_dim<T, Is>...};
}

template <typename T>
constexpr std::array<tensorfuncs::primal::unary::mean_dim_fn<T>, MAX_NDIMS>
get_meanDim() {
    return get_meanDim_impl<T>(std::make_index_sequence<MAX_NDIMS>());
}

/// @brief Returns full table of mean_dim gradient implementations.
template <typename T, size_t... Is>
constexpr std::array<tensorfuncs::adjoint::unary::mean_dim_fn<T>, sizeof...(Is)>
get_meanDim_grad_impl(std::index_sequence<Is...>) {
    return {&tensorfuncs::adjoint::unary::mean_dim<T, Is>...};
}

template <typename T>
constexpr std::array<tensorfuncs::adjoint::unary::mean_dim_fn<T>, MAX_NDIMS>
get_meanDim_grad() {
    return get_meanDim_grad_impl<T>(std::make_index_sequence<MAX_NDIMS>());
}

/// @brief Returns full table of slice operation implementations.
template <typename T, size_t... Is>
constexpr std::array<tensorfuncs::primal::unary::slice_fn<T>, sizeof...(Is)>
get_slice_impl(std::index_sequence<Is...>) {
    return {&tensorfuncs::primal::unary::slice<T, Is>...};
}

template <typename T>
constexpr std::array<tensorfuncs::primal::unary::slice_fn<T>, MAX_NDIMS>
get_slice() {
    return get_slice_impl<T>(std::make_index_sequence<MAX_NDIMS>());
}

/// @brief Returns full table of slice gradient implementations.
template <typename T, size_t... Is>
constexpr std::array<tensorfuncs::adjoint::unary::slice_fn<T>, sizeof...(Is)>
get_slice_grad_impl(std::index_sequence<Is...>) {
    return {&tensorfuncs::adjoint::unary::slice<T, Is>...};
}

template <typename T>
constexpr std::array<tensorfuncs::adjoint::unary::slice_fn<T>, MAX_NDIMS>
get_slice_grad() {
    return get_slice_grad_impl<T>(std::make_index_sequence<MAX_NDIMS>());
}

} // namespace Dispatchers
} // namespace kaad
