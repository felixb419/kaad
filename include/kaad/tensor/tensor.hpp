#pragma once

#include "../scalar.hpp"     // for Scalar
#include "iterator_impl.hpp" // for iterator_impl
#include "tensor_view.hpp"   // for Tensor_view, Tensor_view_mut
#include <cstddef>           // for size_t
#include <cstdint>           // for uint64_t
#include <iostream>          // for ostream, ptrdiff_t
#include <random>            // for mt19937_64
#include <span>              // for span
#include <vector>            // for vector

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
    using difference_type = std::ptrdiff_t;
    using size_type = std::size_t;

    using iterator = iterator_impl<false>;
    using const_iterator = iterator_impl<true>;

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
     * @brief Compute per-dim stride array based on @p shape.
     * @param shape Shape array for wich the strides are computed.
     * @return Vector containing the strides.
     */
    static std::vector<int> compute_stride(std::span<const int> shape);

    /**
     * @brief Compute the size of a element array based on @p shape.
     * @param shape Shape array for which the size is computed.
     * @return Size of the element array suggested by @p shape.
     */
    static size_type compute_size(std::span<const int> shape);

    /**
     * @brief Constructs tensor with shape: [0]
     */
    Tensor();

    /**
     * @brief Constructs a tensor with given @p shape.
     * @note The elements array is initialized ot 0.
     * @param shape Array with the shape of the tensor.
     */
    explicit Tensor(std::span<const int> shape);

    /**
     * @brief Constructs a rank-1 tensor with given @p elements.
     * @param shape Array with the shape of the tensor.
     */
    explicit Tensor(std::span<const Scalar> elements);

    /**
     * @brief Constructs a tensor with given @p shape and @p stride.
     * @note The elements array is initialized ot 0.
     * @param shape Array with the shape of the tensor.
     * @param stride Array with the per-dim strides of the tensor.
     */
    Tensor(std::span<const int> shape, std::span<const int> stride);

    /**
     * @brief Constructs a tensor with given @p shape and @p elements.
     * @param shape Array with the shape of the tensor.
     * @param elements Array with the values of the tensor.
     */
    Tensor(std::span<const int> shape, std::span<const Scalar> elements);

    /**
     * @brief Constructs a tensor with given @p shape, @p stride and @p
     * elements.
     * @param shape Array with the shape of the tensor.
     * @param stride Array with the per-dim strides of the tensor.
     * @param elements Array with the values of the tensor.
     */
    Tensor(std::span<const int> shape, std::span<const int> stride,
           std::span<const Scalar> elements);

    /**
     * @brief Returns a tensor with given shape and filled with @p fill_value.
     * @param shape Shape array for the tensor.
     * @param fill_value Value to fill the element array.
     */
    static Tensor full(std::span<const int> shape, Scalar fill_value);

    /**
     * @brief Returns a tensor with given shape and filled with 0.
     * @param shape Shape array for the tensor.
     */
    static Tensor zeros(std::span<const int> shape);

    /**
     * @brief Returns a tensor with given shape and filled with 1.
     * @param shape Shape array for the tensor.
     */
    static Tensor ones(std::span<const int> shape);

    /**
     * @brief Returns a tensor with given shape and sequentially increasing
     * values.
     * @param shape Shape array for the tensor.
     * @param starting_value The value to start the sequence at.
     */
    static Tensor sequential(std::span<const int> shape,
                             Scalar starting_value = 1);

    /**
     * @brief Returns a tensor with given shape and linearly spaced values.
     * @param shape Shape array for the tensor.
     */
    static Tensor linspace(std::span<const int> shape, Scalar start,
                           Scalar step);

    /**
     * @brief Returns a tensor with given shape and filled with random values.
     * @param shape Shape array for the tensor.
     */
    static Tensor rand(std::span<const int> shape, Scalar min = 0,
                       Scalar max = 1);

    /**
     * @brief Returns a tensor with given shape and filled with random values,
     * sampled from a normal distribution.
     * @param shape Shape array for the tensor.
     * @param mean Mean of the produced values.
     * @param std Standard deviation of the produced values.
     */
    static Tensor randn(std::span<const int> shape, Scalar mean = 0,
                        Scalar std = 1);

    /**
     * @brief Set manual seed for random number generation.
     * @param seed Seed to be used.
     */
    static void manual_seed(uint64_t seed);

    /**
     * @brief Reshapes the tensor.
     * @note Recomputes the stride array for the new shape.
     * @param shape New shape. Must preserve the total number of elements.
     */
    void reshape(std::span<const int> shape);

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
     * @brief Returns the tensor stride array.
     * The stride specifies the memory offset (in elements) between
     * consecutive entries along each dimension.
     * @return Const reference to the stride array.
     */
    const std::vector<int> &stride() const noexcept;

    /**
     * @brief Get an iterator to the begin of the element array.
     * @warning Tensor::iterator is not a contignous iterator (use
     * Tensor::data() for that).
     * @return Iterator to the first element.
     */
    iterator begin();

    /**
     * @brief Get a const iterator to the begin of the element array.
     * @warning Tensor::iterator is not a contignous iterator (use
     * Tensor::data() for that).
     * @return Const iterator to the first element.
     */
    const_iterator begin() const;

    /**
     * @brief Get an iterator to the end of the element array.
     * @warning Tensor::iterator is not a contignous iterator (use
     * Tensor::data() for that).
     * @return Iterator to one past the last element.
     */
    iterator end();

    /**
     * @brief Get a const iterator to the end of the element array.
     * @warning Tensor::iterator is not a contignous iterator (use
     * Tensor::data() for that).
     * @return Const iterator to one past the last element.
     */
    const_iterator end() const;

    /**
     * @brief Get number of elements in the tensor.
     * @return Length of the element array.
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
     * @brief Get a pointer to the element array.
     * @return Pointer to the start of the element array.
     */
    pointer data() noexcept;

    /**
     * @brief Get a const pointer to the element array.
     * @return Const pointer to the start of the element array.
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
