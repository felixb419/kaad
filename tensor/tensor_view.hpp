#pragma once

#include "common.hpp" // for kaad::detail::print_tensor
#include <cstddef>    // for std::size_t
#include <iostream>   // for std::ostream, std::cout
#include <vector>     // for std::vector

namespace kaad {

/**
 * @brief Lightweight immutable view into a tensor's shape, stride, and data.
 * @tparam T The data type stored in the tensor.
 */
template <typename T> struct Tensor_view {
    using value_type = T;
    using reference = value_type &;
    using const_reference = const value_type &;
    using pointer = value_type *;
    using const_pointer = const value_type *;

    using iterator = value_type *;
    using const_iterator = const value_type *;
    using difference_type = std::ptrdiff_t;
    using size_type = std::size_t;

    const int *shape = nullptr;  ///< Pointer to the shape array.
    const int *stride = nullptr; ///< Pointer to the stride array.
    size_type nDims = 0;         ///< Length of the shape and stride arrays.
    const value_type *elements = nullptr; ///< Pointer to the value array.
    size_type len = 0;                    ///< Length of the value array.

    /**
     * @brief Default constructor.
     */
    Tensor_view() {}

    /**
     * @brief Constructs a tensor view.
     * @param shape Pointer to the shape array.
     * @param shape Pointer to the stride array.
     * @param nDims Length of the shape and stride arrays.
     * @param elements Pointer to the value array.
     * @param len Length of the value array.
     */
    Tensor_view(const int *shape, const int *stride, size_type nDims,
                const value_type *elements, size_type len)
        : shape(shape), stride(stride), nDims(nDims), elements(elements),
          len(len) {}

    /**
     * @brief Overloads the stream output operator to print the tensor view.
     *
     * @param stream Output stream.
     * @param view The tensor view to print.
     * @return std::ostream& The updated output stream.
     */
    friend std::ostream &operator<<(std::ostream &stream,
                                    const Tensor_view<value_type> &view) {
        if (view.nDims == 0) {
            std::cout << "[]";
        } else {
            std::vector<int> cords(view.nDims);
            int indent = 0;

            detail::print_tensor(stream, cords, view.shape, view.stride,
                                 view.elements, 0, indent);
        }
        return stream;
    }
};

/**
 * @brief Lightweight mutable view into a tensor's shape, stride, and data.
 * @tparam T The data type stored in the tensor.
 */
template <typename T> struct Tensor_view_mut {
    using value_type = T;
    using reference = value_type &;
    using const_reference = const value_type &;
    using pointer = value_type *;
    using const_pointer = const value_type *;

    using iterator = value_type *;
    using const_iterator = const value_type *;
    using difference_type = std::ptrdiff_t;
    using size_type = std::size_t;

    const int *shape = nullptr;     ///< Pointer to the shape array.
    const int *stride = nullptr;    ///< Pointer to the stride array.
    size_type nDims = 0;            ///< Length of the shape and stride arrays.
    value_type *elements = nullptr; ///< Pointer to the value array.
    size_type len = 0;              ///< Length of the value array.

    /**
     * @brief Default constructor.
     */
    Tensor_view_mut() {}

    /**
     * @brief Constructs a tensor view.
     * @param shape Pointer to the shape array.
     * @param shape Pointer to the stride array.
     * @param nDims Length of the shape and stride arrays.
     * @param elements Pointer to the value array.
     * @param len Length of the value array.
     */
    Tensor_view_mut(const int *shape, const int *stride, size_type nDims,
                    value_type *elements, size_type len)
        : shape(shape), stride(stride), nDims(nDims), elements(elements),
          len(len) {}

    /**
     * @brief Get an immutable view of the Tensor.
     * @return Immutable view (Tensor_view).
     */
    Tensor_view<value_type> make_immutable() {
        return Tensor_view<value_type>(this->shape, this->stride, this->nDims,
                                       this->elements, this->len);
    }

    /**
     * @brief Overloads the stream output operator to print the tensor view.
     * @param stream Output stream.
     * @param view The tensor view to print.
     * @return std::ostream& The updated output stream.
     */
    friend std::ostream &operator<<(std::ostream &stream,
                                    const Tensor_view_mut<value_type> &view) {
        if (view.nDims == 0) {
            std::cout << "[]";
        } else {
            std::vector<int> cords(view.nDims);
            int indent = 0;

            detail::print_tensor(stream, cords, view.shape, view.stride,
                                 view.elements, 0, indent);
        }
        return stream;
    }
};

} // namespace kaad
