#pragma once

#include "gradients.h"    // for batchmatmulGrad, flexBinaryGrad, meanDimGrad
#include "operations.h"   // for batchmatmulOp, flexBinaryOp, meanDimOp
#include <array>          // for array
#include <bits/utility.h> // for index_sequence, make_index_sequence
#include <cstddef>        // for size_t

#ifndef KAAD_MAX_NDIMS
#define KAAD_MAX_NDIMS 5
#endif

using namespace kaad;

namespace kaad {

namespace dispatchers {

/// flexible Op
template <typename T, class Op, std::size_t... Is>
constexpr std::array<flexBinaryOp<T, Op>, sizeof...(Is)>
get_flexOp_impl(std::index_sequence<Is...>) {
    return {&Operations::binary::flexible<T, Op, Is>...};
}

template <typename T, class Op>
constexpr std::array<flexBinaryOp<T, Op>, KAAD_MAX_NDIMS> get_flexOp() {
    return get_flexOp_impl<T, Op>(std::make_index_sequence<KAAD_MAX_NDIMS>());
}

/// flexible Grad
template <typename T, class Grad, std::size_t... Is>
constexpr std::array<flexBinaryGrad<T, Grad>, sizeof...(Is)>
get_flexGrad_impl(std::index_sequence<Is...>) {
    return {&Gradients::binary::flexible<T, Grad, Is>...};
}

template <typename T, class Grad>
constexpr std::array<flexBinaryGrad<T, Grad>, KAAD_MAX_NDIMS> get_flexGrad() {
    return get_flexGrad_impl<T, Grad>(
        std::make_index_sequence<KAAD_MAX_NDIMS>());
}

/// batch matmul Op
template <typename T, std::size_t... Is>
constexpr std::array<batchmatmulOp<T>, sizeof...(Is)>
get_batch_matmul_impl(std::index_sequence<Is...>) {
    return {&Operations::binary::batch_matmul<T, Is>...};
}

template <typename T>
constexpr std::array<batchmatmulOp<T>, KAAD_MAX_NDIMS> get_batch_matmul() {
    return get_batch_matmul_impl<T>(std::make_index_sequence<KAAD_MAX_NDIMS>());
}

/// batch matmul Grad
template <typename T, std::size_t... Is>
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

/// sum dim Op
template <typename T, std::size_t... Is>
constexpr std::array<sumDimOp<T>, sizeof...(Is)>
get_sumDim_impl(std::index_sequence<Is...>) {
    return {&Operations::unary::sum_dim<T, Is>...};
}

template <typename T>
constexpr std::array<sumDimOp<T>, KAAD_MAX_NDIMS> get_sumDim() {
    return get_sumDim_impl<T>(std::make_index_sequence<KAAD_MAX_NDIMS>());
}

/// sum dim Grad
template <typename T, std::size_t... Is>
constexpr std::array<sumDimGrad<T>, sizeof...(Is)>
get_sumDim_grad_impl(std::index_sequence<Is...>) {
    return {&Gradients::unary::sum_dim<T, Is>...};
}

template <typename T>
constexpr std::array<sumDimGrad<T>, KAAD_MAX_NDIMS> get_sumDim_grad() {
    return get_sumDim_grad_impl<T>(std::make_index_sequence<KAAD_MAX_NDIMS>());
}

/// mean dim Op
template <typename T, std::size_t... Is>
constexpr std::array<meanDimOp<T>, sizeof...(Is)>
get_meanDim_impl(std::index_sequence<Is...>) {
    return {&Operations::unary::mean_dim<T, Is>...};
}

template <typename T>
constexpr std::array<meanDimOp<T>, KAAD_MAX_NDIMS> get_meanDim() {
    return get_meanDim_impl<T>(std::make_index_sequence<KAAD_MAX_NDIMS>());
}

/// mean dim Grad
template <typename T, std::size_t... Is>
constexpr std::array<meanDimGrad<T>, sizeof...(Is)>
get_meanDim_grad_impl(std::index_sequence<Is...>) {
    return {&Gradients::unary::mean_dim<T, Is>...};
}

template <typename T>
constexpr std::array<meanDimGrad<T>, KAAD_MAX_NDIMS> get_meanDim_grad() {
    return get_meanDim_grad_impl<T>(std::make_index_sequence<KAAD_MAX_NDIMS>());
}

/// slice Op
template <typename T, std::size_t... Is>
constexpr std::array<sliceOp<T>, sizeof...(Is)>
get_slice_impl(std::index_sequence<Is...>) {
    return {&Operations::unary::slice<T, Is>...};
}

template <typename T>
constexpr std::array<sliceOp<T>, KAAD_MAX_NDIMS> get_slice() {
    return get_slice_impl<T>(std::make_index_sequence<KAAD_MAX_NDIMS>());
}

/// slice Grad
template <typename T, std::size_t... Is>
constexpr std::array<sliceGrad<T>, sizeof...(Is)>
get_slice_grad_impl(std::index_sequence<Is...>) {
    return {&Gradients::unary::slice<T, Is>...};
}

template <typename T>
constexpr std::array<sliceGrad<T>, KAAD_MAX_NDIMS> get_slice_grad() {
    return get_slice_grad_impl<T>(std::make_index_sequence<KAAD_MAX_NDIMS>());
}

} // namespace dispatchers
} // namespace kaad
