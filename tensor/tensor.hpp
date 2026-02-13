#pragma once

#include "../scalar.hpp"    // for Scalar
#include "tensor_view.hpp"  // for kaad::Tensor_view
#include <concepts>         // for concept
#include <cstddef>          // for size_t
#include <cstdint>          // for uint64_t
#include <initializer_list> // for std::initializer_list
#include <iostream>         // for std::ostream
#include <random>           // for std::mt19937_64
#include <ranges>           // for std::ranges
#include <vector>           // for std::vector

namespace kaad {

/**
 * @brief A class representing a multi-dimensional tensor.
 */
class Tensor {
  public:
    using value_type = Scalar;
    using reference = value_type &;
    using const_reference = const value_type &;
    using pointer = value_type *;
    using const_pointer = const value_type *;

    using iterator = value_type *;
    using const_iterator = const value_type *;
    using difference_type = std::ptrdiff_t;
    using size_type = std::size_t;

  private:
    std::vector<int>
        shape_; ///< Vector containing the size of the tensor in each dimension.
    std::vector<int>
        stride_; ///< Vector containing the stride of the tensor (steps needed
                 ///< to move one element in each dimension).
    std::vector<value_type>
        elements_; ///< Vector containing the elements of the Tensor.

    thread_local static inline std::mt19937_64 gen_;
    thread_local static inline bool seeded_ = false;

  public:
    /**
     * @brief Constructs tensor with shape: [0]
     */
    Tensor();

    /**
     * @brief Constructs a tensor with given @p shape (value array stays
     * uninitialized).
     * @param shape Array with the shape of the tensor.
     */
    Tensor(std::span<const int> shape);

    /**
     * @brief Constructs a tensor with given @p shape and @p stride (value array
     * stays uninitialized).
     * @param shape Array with the shape of the tensor.
     * @param stride Array with the per-dim strides of the tensor.
     */
    Tensor(std::span<const int> shape, std::span<const int> stride);

    /**
     * @brief Constructs a tensor with given @p shape and @p elements.
     * @param shape Array with the shape of the tensor.
     * @param elements Array with the values of the tensor.
     */
    Tensor(std::span<const int> shape, std::span<Scalar> elements);

    /**
     * @brief Returns a tensor with given shape and uninitialized values.
     * @param shape Shape array for the tensor.
     */
    static Tensor empty(std::initializer_list<int> shape);

    /**
     * @brief Returns a tensor with given shape and filled with @p fill_value.
     * @param shape Shape array for the tensor.
     * @param fill_value Value to fill the value array.
     */
    static Tensor full(std::initializer_list<int> shape, Scalar fill_value);

    /**
     * @brief Returns a tensor with given shape and filled with 0.
     * @param shape Shape array for the tensor.
     */
    static Tensor zeros(std::initializer_list<int> shape);

    /**
     * @brief Returns a tensor with given shape and filled with 1.
     * @param shape Shape array for the tensor.
     */
    static Tensor ones(std::initializer_list<int> shape);

    /**
     * @brief Returns a tensor with given shape and sequentially increasing
     * values.
     * @param shape Shape array for the tensor.
     * @param starting_value The value to start the sequence at.
     */
    static Tensor sequential(std::initializer_list<int> shape,
                             Scalar starting_value = 1);

    /**
     * @brief Returns a tensor with given shape and linearly spaced values.
     * @param shape Shape array for the tensor.
     */
    static Tensor linspace(std::initializer_list<int> shape, Scalar start,
                           Scalar step);

    /**
     * @brief Returns a tensor with given shape and filled with random values.
     * @param shape Shape array for the tensor.
     */
    static Tensor rand(std::initializer_list<int> shape, Scalar min = 0,
                       Scalar max = 1);

    /**
     * @brief Returns a tensor with given shape and filled with random values,
     * sampled from a normal distribution.
     * @param shape Shape array for the tensor.
     * @param mean Mean of the produced values.
     * @param std Standard deviation of the produced values.
     */
    static Tensor randn(std::initializer_list<int> shape, Scalar mean = 0,
                        Scalar std = 1);

    /**
     * @brief Set manual seed for random number generation.
     * @param seed Seed to be used.
     */
    static void manual_seed(uint64_t seed);

    /**
     * @brief Get number of dimensions of the tensor.
     * @return Length of the shape array.
     */
    size_type rank() const noexcept;

    /**
     * @brief Get immutable reference to the shape array.
     * @return Const reference to shape_.
     */
    const std::vector<int> &shape() const noexcept;

    /**
     * @brief Get immutable reference to the stride array.
     * @return Const reference to stride_.
     */
    const std::vector<int> &stride() const noexcept;

    /**
     * @brief Get an iterator to the begin of the value array.
     * @return Iterator to the first element.
     */
    iterator begin() noexcept;

    /**
     * @brief Get a const iterator to the begin of the value array.
     * @return Const iterator to the first element.
     */
    const_iterator begin() const noexcept;

    /**
     * @brief Get an iterator to the end of the value array.
     * @return Iterator to one past the last element.
     */
    iterator end() noexcept;

    /**
     * @brief Get a const iterator to the end of the value array.
     * @return Const iterator to one past the last element.
     */
    const_iterator end() const noexcept;

    /**
     * @brief Get number of elements in the tensor.
     * @return Length of the value array.
     */
    size_type size() const noexcept;

    /**
     * @brief Checks if tensor has no elements
     * @return True if container is empty, false otherwise.
     */
    bool empty() const noexcept;

    /**
     * @brief Returns a reference to the first element.
     */
    reference front() noexcept;

    /**
     * @brief Returns an immutable reference to the first element.
     */
    const_reference front() const noexcept;

    /**
     * @brief Returns a reference to the last element.
     */
    reference back() noexcept;

    /**
     * @brief Returns an immutable reference to the last element.
     */
    const_reference back() const noexcept;

    /**
     * @brief Get a pointer to the value array.
     * @return Pointer to the start of the value array.
     */
    pointer data() noexcept;

    /**
     * @brief Get a const pointer to the value array.
     * @return Const pointer to the start of the value array.
     */
    const_pointer data() const noexcept;

    /**
     * @brief Creates a view of the tensor.
     * @return A Tensor_view<T> structure representing a non-owning
     * immutable view of the tensor.
     */
    struct Tensor_view view() const noexcept;

    /**
     * @brief Creates a view of the tensor.
     * @return A Tensor_view<T> structure representing a non-owning mutable
     * view of the tensor.
     */
    struct Tensor_view_mut view_mut() noexcept;

    friend class Computation_graph;
    friend class INode;

    template <class Kernel> friend class Node_binary;
    template <class Kernel> friend class Node_binary_flex;
    friend class Node_dot;
    friend class Node_matmul;
    friend class Node_batch_matmul;
    friend class Node_outer;

    template <class Kernel> friend class Node_unary;
    friend class Node_slice;
    friend class Node_transp;
    friend class Node_sum_dim;
    friend class Node_mean;
    friend class Node_mean_dim;
};

/**
 * @brief Stream output operator.
 * @param stream Output stream.
 * @param tensor The tensor to print.
 * @return Reference to the output stream.
 */
std::ostream &operator<<(std::ostream &os, const Tensor &tensor);

} // namespace kaad
