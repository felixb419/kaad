#pragma once

#include <array>
#include <concepts>
#include <kaad/graph/internal/inode.hpp>

namespace kaad {

/// @brief Wildcard struct convertible to anything.
struct Any {
    template <class T> operator T() const;
};

template <class operation>
concept HasOpName = requires {
    requires std::same_as<std::remove_cv_t<decltype(operation::OPERATION_NAME)>,
                          const char *>;
};

template <class operation>
concept HasArity = requires {
    requires std::same_as<std::remove_cv_t<decltype(operation::ARITY)>,
                          std::size_t>;
};

template <class operation>
concept HasMakeResShape =
    HasArity<operation> && (requires(std::array<INode *, operation::ARITY> inputs) {
        { operation::make_res_shape(inputs) } -> std::same_as<Shape>;
    } || 
    // signature with one extra param
    requires(std::array<INode *, operation::ARITY> inputs) {
        { operation::make_res_shape(inputs, Any{}) } -> std::same_as<Shape>;
    } || 
    // signature with two extra params
   requires(std::array<INode *, operation::ARITY> inputs) {
        { operation::make_res_shape(inputs, Any{}, Any{}) } -> std::same_as<Shape>;
    } || 
    // signature with one extra param and return includes strides
   requires(std::array<INode *, operation::ARITY> inputs) {
        { operation::make_res_shape(inputs, Any{}) } -> std::same_as<std::pair<Shape, Strides>>;
    });

template <class operation>
concept HasParams =
    requires(std::array<INode *, operation::ARITY> inputs, INode *result) {
        {
            typename operation::ForwardParams(inputs, result)
        } -> std::same_as<typename operation::ForwardParams>;

        {
            typename operation::BackwardParams(inputs, result)
        } -> std::same_as<typename operation::BackwardParams>;
    } ||
    // signature with one extra param
    requires(std::array<INode *, operation::ARITY> inputs, INode *result) {
        {
            typename operation::ForwardParams(inputs, result, Any{})
        } -> std::same_as<typename operation::ForwardParams>;

        {
            typename operation::BackwardParams(inputs, result, Any{})
        } -> std::same_as<typename operation::BackwardParams>;
    } ||
    // signature with two extra params
    requires(std::array<INode *, operation::ARITY> inputs, INode *result) {
        {
            typename operation::ForwardParams(inputs, result, Any{}, Any{})
        } -> std::same_as<typename operation::ForwardParams>;

        {
            typename operation::BackwardParams(inputs, result, Any{}, Any{})
        } -> std::same_as<typename operation::BackwardParams>;
    };

template <class operation>
concept HasDispatch =
    requires(operation::Dispatch dis) {
        // make sure dispatch has required members
        {
            dis.forward(
                std::declval<const typename operation::ForwardParams &>())
        } -> std::same_as<void>;
        {
            dis.backward(
                std::declval<const typename operation::BackwardParams &>())
        } -> std::same_as<void>;
    } &&
    // make sure there is dispatch function
    (
        requires(std::array<INode *, operation::ARITY> inputs, INode *result) {
            {
                operation::dispatch(inputs, result)
            } -> std::same_as<typename operation::Dispatch>;
        } ||
        // signature with one extra param
        requires(std::array<INode *, operation::ARITY> inputs, INode *result) {
            {
                operation::dispatch(inputs, result, Any{})
            } -> std::same_as<typename operation::Dispatch>;
        } ||
        // signature with two extra params
        requires(std::array<INode *, operation::ARITY> inputs, INode *result) {
            {
                operation::dispatch(inputs, result, Any{}, Any{})
            } -> std::same_as<typename operation::Dispatch>;
        });

template <class operation>
concept Operation =
    HasOpName<operation> && HasArity<operation> && HasMakeResShape<operation> &&
    HasParams<operation> && HasDispatch<operation>;

} // namespace kaad
