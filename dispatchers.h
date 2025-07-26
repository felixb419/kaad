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

/*
flexible Op
*/
template <typename T, class Op, std::size_t... Is>
constexpr std::array<flexBinaryOp<T, Op>, sizeof...(Is)>
get_flexOp_dispatcher_impl(std::index_sequence<Is...>) {
    return {&Operations::flexible<T, Op, Is>...};
}

template <typename T, class Op>
constexpr std::array<flexBinaryOp<T, Op>, KAAD_MAX_NDIMS>
get_flexOp_dispatcher() {
    return get_flexOp_dispatcher_impl<T, Op>(
        std::make_index_sequence<KAAD_MAX_NDIMS>());
}

/*
flexible Grad
*/
template <typename T, class Grad, std::size_t... Is>
constexpr std::array<flexBinaryGrad<T, Grad>, sizeof...(Is)>
get_flexGrad_dispatcher_impl(std::index_sequence<Is...>) {
    return {&Gradients::flexible<T, Grad, Is>...};
}

template <typename T, class Grad>
constexpr std::array<flexBinaryGrad<T, Grad>, KAAD_MAX_NDIMS>
get_flexGrad_dispatcher() {
    return get_flexGrad_dispatcher_impl<T, Grad>(
        std::make_index_sequence<KAAD_MAX_NDIMS>());
}

/*
batch matmul Op
*/
template <typename T, std::size_t... Is>
constexpr std::array<batchmatmulOp<T>, sizeof...(Is)>
get_batch_matmul_dispatcher_impl(std::index_sequence<Is...>) {
    return {&Operations::batch_matmul<T, Is>...};
}

template <typename T>
constexpr std::array<batchmatmulOp<T>, KAAD_MAX_NDIMS>
get_batch_matmul_dispatcher() {
    return get_batch_matmul_dispatcher_impl<T>(
        std::make_index_sequence<KAAD_MAX_NDIMS>());
}

/*
batch matmul Grad
*/
template <typename T, std::size_t... Is>
constexpr std::array<batchmatmulGrad<T>, sizeof...(Is)>
get_batch_matmul_grad_dispatcher_impl(std::index_sequence<Is...>) {
    return {&Gradients::batch_matmul<T, Is>...};
}

template <typename T>
constexpr std::array<batchmatmulGrad<T>, KAAD_MAX_NDIMS>
get_batch_matmul_grad_dispatcher() {
    return get_batch_matmul_grad_dispatcher_impl<T>(
        std::make_index_sequence<KAAD_MAX_NDIMS>());
}

/*
sum dim Op
*/
template <typename T, std::size_t... Is>
constexpr std::array<sumDimOp<T>, sizeof...(Is)>
get_sumDim_dispatcher_impl(std::index_sequence<Is...>) {
    return {&Operations::sum_dim<T, Is>...};
}

template <typename T>
constexpr std::array<sumDimOp<T>, KAAD_MAX_NDIMS> get_sumDim_dispatcher() {
    return get_sumDim_dispatcher_impl<T>(
        std::make_index_sequence<KAAD_MAX_NDIMS>());
}

/*
sum dim Grad
*/
template <typename T, std::size_t... Is>
constexpr std::array<sumDimGrad<T>, sizeof...(Is)>
get_sumDim_grad_dispatcher_impl(std::index_sequence<Is...>) {
    return {&Gradients::sum_dim<T, Is>...};
}

template <typename T>
constexpr std::array<sumDimGrad<T>, KAAD_MAX_NDIMS>
get_sumDim_grad_dispatcher() {
    return get_sumDim_grad_dispatcher_impl<T>(
        std::make_index_sequence<KAAD_MAX_NDIMS>());
}

/*
mean dim Op
*/
template <typename T, std::size_t... Is>
constexpr std::array<meanDimOp<T>, sizeof...(Is)>
get_meanDim_dispatcher_impl(std::index_sequence<Is...>) {
    return {&Operations::mean_dim<T, Is>...};
}

template <typename T>
constexpr std::array<meanDimOp<T>, KAAD_MAX_NDIMS> get_meanDim_dispatcher() {
    return get_meanDim_dispatcher_impl<T>(
        std::make_index_sequence<KAAD_MAX_NDIMS>());
}

/*
mean dim Grad
*/
template <typename T, std::size_t... Is>
constexpr std::array<meanDimGrad<T>, sizeof...(Is)>
get_meanDim_grad_dispatcher_impl(std::index_sequence<Is...>) {
    return {&Gradients::mean_dim<T, Is>...};
}

template <typename T>
constexpr std::array<meanDimGrad<T>, KAAD_MAX_NDIMS>
get_meanDim_grad_dispatcher() {
    return get_meanDim_grad_dispatcher_impl<T>(
        std::make_index_sequence<KAAD_MAX_NDIMS>());
}
