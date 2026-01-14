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
    using reference = T &;
    using const_reference = const T &;
    using pointer = T *;
    using const_pointer = const T *;

    using iterator = T *;
    using const_iterator = const T *;
    using difference_type = std::ptrdiff_t;
    using size_type = std::size_t;

    std::vector<int>
        shape; ///< Vector containing the size of the tensor in each dimension.
    std::vector<int>
        stride; ///< Vector containing the stride of the tensor (steps needed to
                ///< move one element in each dimension).
    std::vector<T> elements; ///< Vector containing the elements of the Tensor.

    /// @brief Default constructor.
    Tensor() {}

    /**
     * @brief Constructs a tensor.
     * @param shape Array with the shape of the tensor.
     * @param shape Array with the values of the tensor.
     */
    template <IntegralRange IR, typename VR>
        requires ValueRange<VR, T>
    explicit Tensor(IR shape, VR elements)
        : shape(std::ranges::begin(shape), std::ranges::end(shape)),
          elements(std::ranges::begin(elements), std::ranges::end(elements)) {
        int len;
        detail::compute_stride(this->stride, len, this->shape);

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
    Tensor(IR shape, IR stride, T fill = 0)
        : shape(std::ranges::begin(shape), std::ranges::end(shape)),
          stride(std::ranges::begin(stride), std::ranges::end(stride)) {
        int len = 1;
        for (int d : shape) {
            len *= d;
        }

        this->elements.resize(len);
        std::fill(this->elements.begin(), this->elements.end(), fill);
    }

    /**
     * @brief Constructs a tensor.
     * @param shape Array with the shape of the tensor.
     * @param fill Value to fill the tensor with.
     */
    template <IntegralRange IR>
    Tensor(IR shape, T fill = 0)
        : shape(std::ranges::begin(shape), std::ranges::end(shape)) {
        int len;
        detail::compute_stride(this->stride, len, this->shape);

        this->elements.resize(len);
        std::fill(this->elements.begin(), this->elements.end(), fill);
    }

    /**
     * @brief Constructs a tensor.
     * @param shape Initializer list with the shape of the tensor.
     * @param fill Value to fill the tensor with.
     */
    Tensor(std::initializer_list<int> shape, T fill = 0)
        : shape(shape.begin(), shape.end()) {
        int len;
        detail::compute_stride(this->stride, len, this->shape);

        this->elements.resize(len);
        std::fill(this->elements.begin(), this->end(), fill);
    }

    Tensor(T scalar) : shape(1), stride(1), elements(1) {
        this->shape[0] = 1;
        this->stride[0] = 0;
        this->elements[0] = scalar;
    }

    /**
     * @brief Get number of dimensions of the tensor.
     * @return Length of the shape array.
     */
    size_type nDims() const { return this->shape.size(); }

    /**
     * @brief Get number of elements in the tensor.
     * @return Length of the value array.
     */
    size_type nElems() const { return this->elements.size(); }

    /**
     * @brief Get a const pointer to the value array.
     * @return Const pointer to the start of the value array.
     */
    const T *data() const { return this->elements.data(); }

    /**
     * @brief Get a const pointer to the end of the value array.
     * @return Const pointer to one past the last element.
     */
    const T *end() const { return &(*this->elements.end()); }

    /**
     * @brief Creates a view of the tensor.
     * @return A Tensor_view<T> structure representing a non-owning immutable
     * view of the tensor.
     */
    struct Tensor_view<T> view() const {
        return Tensor_view<T>(this->shape.data(), this->stride.data(),
                              this->nDims(), this->data(), this->nElems());
    }

    /**
     * @brief Creates a view of the tensor.
     * @return A Tensor_view<T> structure representing a non-owning mutable
     * view of the tensor.
     */
    struct Tensor_view<T> view_mut() {
        return Tensor_view_mut<T>(this->shape.data(), this->stride.data(),
                                  this->nDims(), this->data(), this->nElems());
    }

    /**
     * @brief Stream output operator.
     *
     * Prints the tensor in a readable format.
     *
     * @param stream Output stream.
     * @param tensor The tensor to print.
     * @return Reference to the output stream.
     */
    friend std::ostream &operator<<(std::ostream &os, const Tensor<T> &tensor) {
        if (tensor.nDims() == 0) {
            os << "[]";
        } else {
            std::vector<int> cords(tensor.nDims());
            int indent = 0;

            detail::print_tensor(os, cords, tensor.shape, tensor.stride,
                                 tensor.elements, 0, indent);
        }
        return os;
    }
};
} // namespace kaad
