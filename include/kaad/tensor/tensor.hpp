#pragma once

#include <cstddef>                       // for ptrdiff_t, size_t
#include <cstdint>                       // for uint64_t
#include <iostream>                      // for ostream
#include <kaad/mutability_enum.hpp>      // for MUTABILITY
#include <kaad/scalar.hpp>               // for Scalar
#include <kaad/tensor/iterator_impl.hpp> // for IteratorImpl
#include <kaad/tensor/tensor_types.hpp>  // for ShapeView, StrideView, Stride
#include <random>                        // for random_device, mt19937_64
#include <span>                          // for span
#include <vector>                        // for vector

namespace kaad {

template <MUTABILITY M> struct TensorView;

using TensorViewConst = TensorView<IMMUTABLE>;
using TensorViewMut = TensorView<MUTABLE>;

/**
 * @brief A class representing a multi-dimensional tensor.
 *
 * This class owns all of its underlying memory and provides full-value
 * semantics (copy, move, and assignment). It includes standard STL-style
 * type aliases and member functions (e.g., iterators, size(), data()).
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

    using iterator = IteratorImpl<MUTABLE>;
    using const_iterator = IteratorImpl<IMMUTABLE>;

  private:
    Shape shape_;   ///< Vector containing the size of the tensor
                    ///< in each dimension.
    Stride stride_; ///< Vector containing the stride of the tensor (steps
                    ///< needed to move one element in each dimension).
    std::vector<value_type>
        elements_; ///< Vector containing the elements of the Tensor.

    /// @internal
    static std::mt19937_64 &get_rng() {
        thread_local static std::mt19937_64 rng{std::random_device{}()};
        return rng;
    }

  public:
    /// @return Stride array based on @p shape.
    static Stride compute_stride(ShapeView shape);

    /// @return Number of elements based on @p shape.
    static size_type compute_size(ShapeView shape);

    /// @brief Constructs 0-rank tensor initialized to 0.
    Tensor();

    /**
     * @brief Constructs a tensor with given @p shape.
     * @note The elements array is initialized ot 0.
     * @param shape Dimensions of the tensor.
     */
    explicit Tensor(ShapeView shape);

    /// @brief Constructs a rank-1 tensor with given @p elements.
    /// @param elements Elements of the tensor.
    explicit Tensor(std::span<const value_type> elements);

    /**
     * @brief Constructs a tensor with given @p shape and @p stride.
     * @note The elements array is initialized to 0.
     * @param shape Dimensions of the tensor.
     * @param stride Per-dim strides of the tensor.
     */
    Tensor(ShapeView shape, StrideView stride);

    /**
     * @brief Constructs a tensor with given @p shape and @p elements.
     * @param shape Dimensions of the tensor.
     * @param elements Elements of the tensor.
     */
    Tensor(ShapeView shape, std::span<const value_type> elements);

    /**
     * @brief Constructs a tensor with given @p shape, @p stride and @p
     * elements.
     * @param shape Dimensions of the tensor.
     * @param stride Per-dim strides of the tensor.
     * @param elements Elements of the tensor.
     */
    Tensor(ShapeView shape, StrideView stride,
           std::span<const value_type> elements);

    /**
     * @brief Returns a tensor with given shape and filled with @p
     * fill_value.
     * @param shape Dimensions of the tensor.
     * @param fill_value Value to fill the element array.
     */
    static Tensor full(ShapeView shape, value_type fill_value);

    /// @brief Returns a tensor with given shape and filled with 0.
    /// @param shape Dimensions of the tensor.
    static Tensor zeros(ShapeView shape);

    /// @brief Returns a tensor with given shape and filled with 1.
    /// @param shape Dimensions of the tensor.
    static Tensor ones(ShapeView shape);

    /**
     * @brief Returns a tensor with given shape and sequentially increasing
     * values.
     * @param shape Dimensions of the tensor.
     * @param starting_value The value to start the sequence at.
     */
    static Tensor sequential(ShapeView shape, value_type starting_value = 1);

    /// @brief Returns a tensor with given shape and linearly spaced values.
    /// @param shape Dimensions of the tensor.
    static Tensor linspace(ShapeView shape, value_type start, value_type step);

    /**
     * @brief Returns a tensor with given shape and filled with random values.
     * @param shape Dimensions of the tensor.
     * @param min Minimum random value.
     * @param min Maximum random value.
     */
    static Tensor rand(ShapeView shape, value_type min = 0, value_type max = 1);

    /**
     * @brief Returns a tensor with given shape and filled with normally
     * distrubted random values.
     * @param shape Dimensions of the tensor.
     * @param mean Mean of the produced values.
     * @param std Standard deviation of the produced values.
     */
    static Tensor randn(ShapeView shape, value_type mean = 0,
                        value_type std = 1);

    /**
     * @brief Set manual seed for random number generation.
     * @note If this function isnt called std::random_device is used
     * instead.
     * @note Only affects generation in current thread.
     * @param seed Seed to be used.
     */
    static void rng_seed(uint64_t seed);

    /// @brief Get rank of the tensor.
    /// @return Length of the shape array.
    [[nodiscard]] size_type rank() const noexcept;

    /// @brief Get shape of the tensor.
    /// @return Read-only span representing the dimensions of the tensor.
    [[nodiscard]] ShapeView shape() const noexcept;

    /// @brief Get strides of the tensor.
    /// @return Read-only span representing the stride array.
    [[nodiscard]] StrideView stride() const noexcept;

    /**
     * @brief Get the elements of the tensor.
     * @note If the tensor has been transposed the physical and logical
     * order of elements can differ.
     * @return Read/Write span representing the elements.
     */
    [[nodiscard]] std::span<value_type> elements() noexcept;

    /**
     * @copybrief Tensor::elements()
     * @note If the tensor has been transposed the physical and logical
     * order of elements can differ.
     * @return Read-only span representing the elements.
     */
    [[nodiscard]] std::span<const value_type> elements() const noexcept;

    /**
     * @brief Returns an iterator to the first logical element.
     * @note Iterates in logical (transposed) order; use @c data() for
     * contiguous raw access.
     * @return Read/write iterator to the first element.
     */
    [[nodiscard]] iterator begin() noexcept;

    /**
     * @copybrief Tensor::begin()
     * @note Iterates in logical (transposed) order; use @c data() for
     * contiguous raw access.
     * @return Read iterator to the first element.
     */
    [[nodiscard]] const_iterator begin() const noexcept;

    /**
     * @brief Returns an iterator to one past the last logical element.
     * @note Iterates in logical (transposed) order; use @c data() for
     * contiguous raw access.
     * @return Read/write iterator to one past the last element.
     */
    [[nodiscard]] iterator end() noexcept;

    /**
     * @copybrief Tensor::end()
     * @note Iterates in logical (transposed) order; use @c data() for
     * contiguous raw access.
     * @return Read iterator to one past the last element.
     */
    [[nodiscard]] const_iterator end() const noexcept;

    /// @brief Get number of elements in the tensor.
    /// @return Length of the element array.
    [[nodiscard]] size_type size() const noexcept;

    /// @brief Checks if tensor has elements.
    /// @return True if elements is empty, false otherwise.
    [[nodiscard]] bool empty() const noexcept;

    /// @return Reference to the first element.
    [[nodiscard]] reference front() noexcept;

    /// @return Immutable reference to the first element.
    [[nodiscard]] const_reference front() const noexcept;

    /// @return Reference to the first element.
    [[nodiscard]] reference back() noexcept;

    /// @return Reference to the last element.
    [[nodiscard]] const_reference back() const noexcept;

    /**
     * @brief Returns a pointer to the underlying contiguous storage.
     * @note If the tensor is logically transposed, the logical indexing
     * order does not match the physical memory layout (use Tensor::begin
     * then).
     * @return Pointer to the first element in memory.
     */
    [[nodiscard]] pointer data() noexcept;

    /*
     * @copybrief Tensor::data()
     * @note If the tensor is logically transposed, the logical indexing
     * order does not match the physical memory layout (use Tensor::begin
     * then).
     * @return Const pointer to the first element in memory.
     */
    [[nodiscard]] const_pointer data() const noexcept;

    /// @brief Creates a non-owning immutable view of the tensor.
    /// @return A TensorViewConst object.
    [[nodiscard]] TensorViewConst view() const noexcept;

    /// @brief Creates a non-owning mutable view of the tensor.
    /// @return A TensorViewMut object.
    [[nodiscard]] TensorViewMut view_mut() noexcept;

    friend class Graph;
    friend class INode;

    template <class Kernel> friend class NodeBinary;
    template <class Kernel> friend class NodeBinaryFlex;
    friend class NodeDot;
    friend class NodeMatmul;
    friend class NodeBatchMatmul;
    friend class NodeOuter;

    template <class Kernel> friend class NodeUnary;
    friend class NodeSlice;
    friend class NodeTransp;
    friend class NodeSumDim;
    friend class NodeMean;
    friend class NodeMeanDim;
};

/// @brief Prints @p tensor to @p stream.
std::ostream &operator<<(std::ostream &stream, const Tensor &tensor);

} // namespace kaad
