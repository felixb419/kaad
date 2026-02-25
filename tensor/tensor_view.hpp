#pragma once

#include "../scalar.hpp" // for Scalar
#include <cstddef>       // for std::size_t
#include <iostream>      // for std::ostream, std::cout

namespace kaad {

/**
 * @brief Lightweight immutable view into a tensor's shape, stride, and data.
 */
struct Tensor_view {
    using value_type = Scalar;
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
    size_type rank = 0;          ///< Length of the shape and stride arrays.
    const value_type *elements = nullptr; ///< Pointer to the element array.
    size_type len = 0;                    ///< Length of the element array.

    /**
     * @brief Default constructor.
     */
    Tensor_view() noexcept;

    /**
     * @brief Constructs a tensor view.
     * @param shape Pointer to the shape array.
     * @param shape Pointer to the stride array.
     * @param rank Length of the shape and stride arrays.
     * @param elements Pointer to the element array.
     * @param len Length of the element array.
     */
    Tensor_view(const int *shape, const int *stride, size_type rank,
                const value_type *elements, size_type len) noexcept;
};

/**
 * @brief Overloads the stream output operator to print the tensor view.
 *
 * @param stream Output stream.
 * @param view The tensor view to print.
 * @return std::ostream& The updated output stream.
 */
static inline std::ostream &operator<<(std::ostream &os,
                                       const Tensor_view &view);

/**
 * @brief Lightweight mutable view into a tensor's shape, stride, and data.
 */
struct Tensor_view_mut {
    using value_type = Scalar;
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
    size_type rank = 0;             ///< Length of the shape and stride arrays.
    value_type *elements = nullptr; ///< Pointer to the element array.
    size_type len = 0;              ///< Length of the element array.

    /**
     * @brief Default constructor.
     */
    Tensor_view_mut() noexcept;

    /**
     * @brief Constructs a tensor view.
     * @param shape Pointer to the shape array.
     * @param shape Pointer to the stride array.
     * @param rank Length of the shape and stride arrays.
     * @param elements Pointer to the element array.
     * @param len Length of the element array.
     */
    Tensor_view_mut(const int *shape, const int *stride, size_type rank,
                    value_type *elements, size_type len) noexcept;

    /**
     * @brief Get an immutable view of the Tensor.
     * @return Immutable view (Tensor_view).
     */
    Tensor_view make_immutable() noexcept;
};

/**
 * @brief Overloads the stream output operator to print the tensor view.
 * @param stream Output stream.
 * @param view The tensor view to print.
 * @return std::ostream& The updated output stream.
 */
static inline std::ostream &operator<<(std::ostream &os,
                                       const Tensor_view_mut &view);

} // namespace kaad
