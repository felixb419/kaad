#pragma once

#include <algorithm>        // for std::copy, std::max, std::fill
#include <concepts>         // for concept
#include <cstddef>          // for size_t
#include <initializer_list> // for std::initializer_list
#include <iostream>         // for std::ostream
#include <ranges>           // for std::ranges
#include <stdexcept>        // for std::invalid_argument
#include <vector>           // for std::vector

#include "common.hpp"      // for kaad::detail::print_tensor
#include "tensor_view.hpp" // for kaad::Tensor_view

namespace kaad {

namespace detail {
static void compute_stride(std::vector<int> &stride, int &len,
                           const std::vector<int> &shape) {
    stride.resize(shape.size());

    len = 1;
    int i = shape.size() - 1;

    len *= shape[i];
    stride[i--] = 1;

    for (; i >= 0; i--) {
        len *= shape[i];

        stride[i] = shape[i + 1] * stride[i + 1];
    }

    for (size_t i = 0; i < stride.size(); i++) {
        if (shape[i] <= 1) {
            stride[i] = 0;
        }
    }
}
} // namespace detail

template <typename R>
concept IntegralRange =
    std::ranges::input_range<R> && std::integral<std::ranges::range_value_t<R>>;

template <typename R, typename T>
concept ValueRange = std::ranges::input_range<R> &&
                     std::same_as<std::ranges::range_value_t<R>, T>;

/**
 * @brief A class representing a multi-dimensional tensor.
 *
 * This class supports creation, copying, moving, and indexing of tensors,
 * and provides utilities for memory management and shape/stride calculation.
 *
 * @tparam T The type of the values stored in the tensor.
 */
template <typename T> class Tensor {
  public:
    using value_type = T;
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

  public:
    /// @brief Default constructor.
    Tensor() {}

    /**
     * @brief Constructs a tensor.
     * @param shape Array with the shape of the tensor.
     * @param shape Array with the values of the tensor.
     */
    template <IntegralRange IR, typename VR>
        requires ValueRange<VR, value_type>
    explicit Tensor(IR shape, VR elements)
        : shape_(std::ranges::begin(shape), std::ranges::end(shape)),
          elements_(std::ranges::begin(elements), std::ranges::end(elements)) {
        int len;
        detail::compute_stride(this->stride_, len, this->shape_);

        if (len != elements.size()) {
            throw std::invalid_argument(
                "length suggested by shape and length of val dont match");
        }
    }

    /**
     * @brief Constructs a tensor.
     * @param shape Array with the shape of the tensor.
     * @param fill Value to fill the tensor with.
     */
    template <IntegralRange IR>
    Tensor(IR shape, IR stride, value_type fill = 0)
        : shape_(std::ranges::begin(shape), std::ranges::end(shape)),
          stride_(std::ranges::begin(stride), std::ranges::end(stride)) {
        int len = 1;
        for (int d : shape) {
            len *= d;
        }

        this->elements_.resize(len);
        std::fill(this->elements_.begin(), this->elements_.end(), fill);
    }

    /**
     * @brief Constructs a tensor.
     * @param shape Array with the shape of the tensor.
     * @param fill Value to fill the tensor with.
     */
    template <IntegralRange IR>
    Tensor(IR shape, value_type fill = 0)
        : shape_(std::ranges::begin(shape), std::ranges::end(shape)) {
        int len;
        detail::compute_stride(this->stride_, len, this->shape_);

        this->elements_.resize(len);
        std::fill(this->elements_.begin(), this->elements_.end(), fill);
    }

    /**
     * @brief Constructs a tensor.
     * @param shape Initializer list with the shape of the tensor.
     * @param fill Value to fill the tensor with.
     */
    Tensor(std::initializer_list<int> shape, value_type fill = 0)
        : shape_(shape.begin(), shape.end()) {
        int len;
        detail::compute_stride(this->stride_, len, this->shape_);

        this->elements_.resize(len);
        std::fill(this->elements_.begin(), this->elements_.end(), fill);
    }

    Tensor(value_type scalar) : shape_(1), stride_(1), elements_(1) {
        this->shape_[0] = 1;
        this->stride_[0] = 0;
        this->elements_[0] = scalar;
    }

    /**
     * @brief Get number of dimensions of the tensor.
     * @return Length of the shape array.
     */
    size_type nDims() const noexcept {
        return static_cast<size_type>(this->shape_.size());
    }

    /**
     * @brief Get immutable reference to the shape array.
     * @return Const reference to shape_.
     */
    const std::vector<int> &shape() const noexcept { return this->shape_; }

    /**
     * @brief Get immutable pointer to the begin of shape array.
     * @return Const pointer to the first element of shape_.
     */
    const int *shape_begin() const noexcept { return this->shape_.data(); }

    /**
     * @brief Get immutable pointer to the end of shape array.
     * @return Const pointer one past the shape array.
     */
    const int *shape_end() const noexcept {
        return this->shape_.data() + this->shape_.size();
    }

    /**
     * @brief Get immutable reference to the stride array.
     * @return Const reference to stride_.
     */
    const std::vector<int> &stride() const noexcept { return this->stride_; }

    /**
     * @brief Get immutable pointer to the begin of stride array.
     * @return Const pointer to the first element of stride_.
     */
    const int *stride_begin() const noexcept { return this->stride_.data(); }

    /**
     * @brief Get immutable pointer to the end of stride array.
     * @return Const pointer one past the stride array.
     */
    const int *stride_end() const noexcept {
        return this->stride_.data() + this->stride_.size();
    }

    /**
     * @brief Get a const iterator to the begin of the value array.
     * @return Const iterator to the first element.
     */
    const_iterator begin() const noexcept {
        return static_cast<const_iterator>(this->elements_.data());
    }

    /**
     * @brief Get a const iterator to the end of the value array.
     * @return Const iterator to one past the last element.
     */
    const_iterator end() const noexcept {
        return static_cast<const_iterator>(this->elements_.data() +
                                           this->elements_.size());
    }

    /**
     * @brief Get number of elements in the tensor.
     * @return Length of the value array.
     */
    size_type size() const noexcept {
        return static_cast<size_type>(this->elements_.size());
    }

    /**
     * @brief Checks if tensor has no elements
     * @return True if container is empty, false otherwise.
     */
    bool empty() const noexcept { return this->elements_.empty(); }

    /**
     * @brief Gets an element based off 1d index.
     * @param 1d index
     * @return Element at @p i.
     */
    const_reference operator[](size_type i) const noexcept {
        return static_cast<const_reference>(this->elements_[i]);
    }

    /**
     * @brief Returns a reference to the first element.
     */
    const_reference front() const noexcept {
        return static_cast<const_reference>(this->elements_.front());
    }

    /**
     * @brief Returns a reference to the last element.
     */
    const_reference back() const noexcept {
        return static_cast<const_reference>(this->elements_.back());
    }

    /**
     * @brief Get a const pointer to the value array.
     * @return Const pointer to the start of the value array.
     */
    const_pointer data() const noexcept {
        return static_cast<const_pointer>(this->elements_.data());
    }

    /**
     * @brief Creates a view of the tensor.
     * @return A Tensor_view<T> structure representing a non-owning immutable
     * view of the tensor.
     */
    struct Tensor_view<value_type> view() const {
        return Tensor_view(this->shape_.data(), this->stride_.data(),
                           this->nDims(), this->data(), this->size());
    }

    /**
     * @brief Creates a view of the tensor.
     * @return A Tensor_view<T> structure representing a non-owning mutable
     * view of the tensor.
     */
    struct Tensor_view view_mut() {
        return Tensor_view_mut(this->shape_.data(), this->stride_.data(),
                               this->nDims(), this->data(), this->size());
    }

    /**
     * @brief Stream output operator.
     * @param stream Output stream.
     * @param tensor The tensor to print.
     * @return Reference to the output stream.
     */
    friend std::ostream &operator<<(std::ostream &os, const Tensor &tensor) {
        if (tensor.nDims() == 0 || tensor.size() == 0) {
            os << "[]";
        } else {
            std::vector<int> cords(tensor.nDims());
            int indent = 0;

            detail::print_tensor(os, cords, tensor.shape_, tensor.stride_,
                                 tensor.elements_, 0, indent);
        }
        return os;
    }

    template <typename U> friend class Computation_graph;
    template <typename U> friend class INode;

    template <typename U, class Kernel> friend class Node_binary;
    template <typename U, class Kernel> friend class Node_binary_flex;
    template <typename U> friend class Node_matmul;
    template <typename U> friend class Node_batch_matmul;

    template <typename U, class Kernel> friend class Node_unary;
    template <typename U> friend class Node_slice;
    template <typename U> friend class Node_transp;
    template <typename U> friend class Node_sum_dim;
    template <typename U> friend class Node_mean;
    template <typename U> friend class Node_mean_dim;
};
} // namespace kaad
