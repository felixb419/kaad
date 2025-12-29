#pragma once

#include <algorithm> // for std::copy, std::max, std::fill
#include <concepts>  // for concept
#include <cstddef>   // for size_t
#include <iostream>  // for std::operator<<, std::ostream, std::cout, std::end
#include <ranges>    // for std::ranges
#include <stdexcept> // for std::invalid_argument
#include <vector>    // for std::vector

#include "common.hpp" // for kaad::detail::print_tensor
#include "tView.hpp"  // for kaad::tView

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
    std::vector<int>
        shape; ///< Vector containing the size of the tensor in each dimension.
    std::vector<int>
        stride; ///< Vector containing the stride of the tensor (steps needed to
                ///< move one element in each dimension).
    std::vector<T> val; ///< Vector containing the elements of the Tensor.

    /// @brief Default constructor.
    Tensor() {}

    /**
     * @brief Constructs a tensor.
     * @param shape Array with the shape of the tensor.
     * @param shape Array with the values of the tensor.
     */
    template <IntegralRange IR, typename VR>
        requires ValueRange<VR, T>
    explicit Tensor(IR shape, VR val)
        : shape(std::ranges::begin(shape), std::ranges::end(shape)),
          val(std::ranges::begin(val), std::ranges::end(val)) {
        int len;
        detail::compute_stride(this->stride, len, this->shape);

        if (len != val.size()) {
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
    Tensor(IR shape, T fill = 0)
        : shape(std::ranges::begin(shape), std::ranges::end(shape)) {
        int len;
        detail::compute_stride(this->stride, len, this->shape);

        this->val.resize(len);
        std::fill(this->val.begin(), this->val.end(), fill);
    }

    size_t nDims() const { return this->shape.size(); }

    /**
     * @brief Creates a view of the tensor.
     *
     * @return A tView<T> structure representing a non-owning view of the
     * tensor.
     */
    struct tView<T> view() const {
        return tView<T>(this->shape, this->stride, this->val);
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
    friend std::ostream &operator<<(std::ostream &os, Tensor<T> tensor) {
        if (tensor.nDims() == 0) {
            std::cout << "[]";
        } else {
            int *cords = new int[tensor.nDims()];
            std::fill(cords, cords + tensor.nDims(), 0);
            int indent = 0;

            detail::print_tensor(os, cords, tensor.shape.data(),
                                 tensor.stride.data(), tensor.nDims(),
                                 tensor.val.data(), 0, indent);

            delete[] cords;
        }
        return os;
    }
};
} // namespace kaad
