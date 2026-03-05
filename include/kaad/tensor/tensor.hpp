#pragma once

#include "../scalar.hpp"    // for Scalar
#include "tensor_view.hpp"  // for Tensor_view, Tensor_view_mut
#include <algorithm>        // for equal, fill
#include <cstddef>          // for size_t
#include <cstdint>          // for uint64_t
#include <initializer_list> // for initializer_list
#include <iostream>         // for ostream, ptrdiff_t
#include <iterator>         // for bidirectional_iterator_tag
#include <random>           // for mt19937_64
#include <span>             // for span
#include <type_traits>      // for conditional_t
#include <vector>           // for vector

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

    template <bool isConst> class iterator_impl {
      private:
        using Tensor_reference =
            std::conditional_t<isConst, const Tensor &, Tensor &>;
        Tensor_reference origin_;       ///< Origin tensor of the iterator.
        std::vector<int> cords_;        ///< Per-dim coordinates of the element.
        const std::vector<int> &shape_; ///< Shape of the tensor.
        const std::vector<int> &stride_; ///< Stride array of the tensor.

      public:
        using iterator_concept = std::bidirectional_iterator_tag;
        using iterator_category = std::bidirectional_iterator_tag;
        using value_type = Tensor::value_type;
        using difference_type = Tensor::difference_type;
        using pointer =
            std::conditional_t<isConst, Tensor::const_pointer, Tensor::pointer>;
        using reference = std::conditional_t<isConst, Tensor::const_reference,
                                             Tensor::reference>;

        iterator_impl(Tensor_reference &origin, std::vector<int> &&cords)
            : origin_(origin), cords_(cords), shape_(origin_.shape()),
              stride_(origin_.stride()) {}

        const Tensor &origin() { return this->origin_; }

        reference operator*() const {
            size_t idx = 0;
            for (size_t i = 0; i < this->origin_.rank(); i++) {
                idx += this->cords_[i] * this->stride_[i];
            }
            return this->origin_.data()[idx];
        }

        iterator_impl &operator++() {
            int rank = static_cast<int>(this->origin_.rank() - 1);

            this->cords_[rank]++;

            while (this->cords_[rank] >= this->shape_[rank]) {

                this->cords_[rank] = 0;
                rank--;
                if (rank >= 0) {
                    this->cords_[rank]++;

                } else {
                    // increment every cord but the last, so iterator points one
                    // past end and return.
                    std::copy(this->shape_.begin(), this->shape_.end(),
                              this->cords_.begin());
                    for (size_t i = 0; i < this->origin_.rank() - 1; i++) {
                        this->cords_[i]--;
                    }
                    break;
                }
            }

            return *this;
        }

        iterator_impl operator++(int) {
            iterator_impl old = *this;
            ++(*this);
            return old;
        }

        iterator_impl &operator--() {
            int rank = static_cast<int>(this->origin_.rank() - 1);

            this->cords_[rank]--;

            while (this->cords_[rank] < 0) {

                this->cords_[rank] = this->shape_[rank] - 1;
                if (rank >= 0) {
                    rank--;
                    this->cords_[rank]--;

                } else {
                    // set cords to all 0 and return
                    std::fill(this->cords_.begin(), this->cords_.end(), 0);
                    break;
                }
            }

            return *this;
        }

        iterator_impl operator--(int) {
            iterator_impl old = *this;
            --(*this);
            return old;
        }

        bool operator==(const iterator_impl &other) const {
            return (&this->origin_ == &other.origin_) &&
                   std::equal(this->cords_.begin(), this->cords_.end(),
                              other.cords_.begin());
        }

        bool operator!=(const iterator_impl &other) const {
            return !(*this == other);
        }
    };

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
    static std::size_t compute_size(std::span<const int> shape);

    /**
     * @brief Constructs tensor with shape: [0]
     */
    Tensor();

    /**
     * @brief Constructs a tensor with given @p shape.
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
     * @brief Returns a tensor with given shape and uninitialized values.
     * @param shape Shape array for the tensor.
     */
    static Tensor empty(std::span<const int> shape);

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
