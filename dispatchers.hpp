#pragma once

#include "tensorfuncs/gradients.hpp"  // for batchmatmulGrad, flexBinaryGrad, meanDimGrad
#include "tensorfuncs/operations.hpp" // for batchmatmulOp, flexBinaryOp, meanDimOp
#include <array>          // for std::array
#include <cstddef>        // for size_t
#include <utility>        // for std::index_sequence, std::make_index_sequence

#ifndef KAAD_MAX_NDIMS
#define KAAD_MAX_NDIMS 5
#endif

using namespace kaad;

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
 * for all valid dimensions up to `KAAD_MAX_NDIMS`. This enables efficient and
 * modular implementations of flexible tensor operations and their gradients.
 *
 * Typical users do not need to call these functions directly unless
 * implementing new operators.
 */
namespace Dispatchers {

/// @brief Returns full table of flexible binary operation implementations.
template <typename T, class Op, size_t... Is>
constexpr std::array<flexBinaryOp<T, Op>, sizeof...(Is)>
get_flexOp_impl(std::index_sequence<Is...>) {
    return {&Operations::binary::flexible<T, Op, Is>...};
}

template <typename T, class Op>
constexpr std::array<flexBinaryOp<T, Op>, KAAD_MAX_NDIMS> get_flexOp() {
    return get_flexOp_impl<T, Op>(std::make_index_sequence<KAAD_MAX_NDIMS>());
}

/// @brief Returns full table of flexible binary gradient implementations.
template <typename T, class Grad, size_t... Is>
constexpr std::array<flexBinaryGrad<T, Grad>, sizeof...(Is)>
get_flexGrad_impl(std::index_sequence<Is...>) {
    return {&Gradients::binary::flexible<T, Grad, Is>...};
}

template <typename T, class Grad>
constexpr std::array<flexBinaryGrad<T, Grad>, KAAD_MAX_NDIMS> get_flexGrad() {
    return get_flexGrad_impl<T, Grad>(
        std::make_index_sequence<KAAD_MAX_NDIMS>());
}

/// @brief Returns full table of batch matmul operation implementations.
template <typename T, size_t... Is>
constexpr std::array<batchmatmulOp<T>, sizeof...(Is)>
get_batch_matmul_impl(std::index_sequence<Is...>) {
    return {&Operations::binary::batch_matmul<T, Is>...};
}

template <typename T>
constexpr std::array<batchmatmulOp<T>, KAAD_MAX_NDIMS> get_batch_matmul() {
    return get_batch_matmul_impl<T>(std::make_index_sequence<KAAD_MAX_NDIMS>());
}

/// @brief Returns full table of batch matmul gradient implementations.
template <typename T, size_t... Is>
constexpr std::array<batchmatmulGrad<T>, sizeof...(Is)>
get_batch_matmul_grad_impl(std::index_sequence<Is...>) {
    return {&Gradients::binary::batch_matmul<T, Is>...};
}

template <typename T>
constexpr std::array<batchmatmulGrad<T>, KAAD_MAX_NDIMS>
get_batch_matmul_grad() {
    return get_batch_matmul_grad_impl<T>(
        std::make_index_sequence<KAAD_MAX_NDIMS>());
}

/// @brief Returns full table of sum_dim operation implementations.
template <typename T, size_t... Is>
constexpr std::array<sumDimOp<T>, sizeof...(Is)>
get_sumDim_impl(std::index_sequence<Is...>) {
    return {&Operations::unary::sum_dim<T, Is>...};
}

template <typename T>
constexpr std::array<sumDimOp<T>, KAAD_MAX_NDIMS> get_sumDim() {
    return get_sumDim_impl<T>(std::make_index_sequence<KAAD_MAX_NDIMS>());
}

/// @brief Returns full table of sum_dim gradient implementations.
template <typename T, size_t... Is>
constexpr std::array<sumDimGrad<T>, sizeof...(Is)>
get_sumDim_grad_impl(std::index_sequence<Is...>) {
    return {&Gradients::unary::sum_dim<T, Is>...};
}

template <typename T>
constexpr std::array<sumDimGrad<T>, KAAD_MAX_NDIMS> get_sumDim_grad() {
    return get_sumDim_grad_impl<T>(std::make_index_sequence<KAAD_MAX_NDIMS>());
}

/// @brief Returns full table of mean_dim operation implementations.
template <typename T, size_t... Is>
constexpr std::array<meanDimOp<T>, sizeof...(Is)>
get_meanDim_impl(std::index_sequence<Is...>) {
    return {&Operations::unary::mean_dim<T, Is>...};
}

template <typename T>
constexpr std::array<meanDimOp<T>, KAAD_MAX_NDIMS> get_meanDim() {
    return get_meanDim_impl<T>(std::make_index_sequence<KAAD_MAX_NDIMS>());
}

/// @brief Returns full table of mean_dim gradient implementations.
template <typename T, size_t... Is>
constexpr std::array<meanDimGrad<T>, sizeof...(Is)>
get_meanDim_grad_impl(std::index_sequence<Is...>) {
    return {&Gradients::unary::mean_dim<T, Is>...};
}

template <typename T>
constexpr std::array<meanDimGrad<T>, KAAD_MAX_NDIMS> get_meanDim_grad() {
    return get_meanDim_grad_impl<T>(std::make_index_sequence<KAAD_MAX_NDIMS>());
}

/// @brief Returns full table of slice operation implementations.
template <typename T, size_t... Is>
constexpr std::array<sliceOp<T>, sizeof...(Is)>
get_slice_impl(std::index_sequence<Is...>) {
    return {&Operations::unary::slice<T, Is>...};
}

template <typename T>
constexpr std::array<sliceOp<T>, KAAD_MAX_NDIMS> get_slice() {
    return get_slice_impl<T>(std::make_index_sequence<KAAD_MAX_NDIMS>());
}

/// @brief Returns full table of slice gradient implementations.
template <typename T, size_t... Is>
constexpr std::array<sliceGrad<T>, sizeof...(Is)>
get_slice_grad_impl(std::index_sequence<Is...>) {
    return {&Gradients::unary::slice<T, Is>...};
}

template <typename T>
constexpr std::array<sliceGrad<T>, KAAD_MAX_NDIMS> get_slice_grad() {
    return get_slice_grad_impl<T>(std::make_index_sequence<KAAD_MAX_NDIMS>());
}

} // namespace Dispatchers
} // namespace kaad
