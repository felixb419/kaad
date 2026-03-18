#pragma once

#include <cstddef>                // for size_t
#include <iostream>               // for ostream, ptrdiff_t
#include <kaad/scalar.hpp>        // for Scalar
#include <kaad/tensor/tensor.hpp> // for Shape, Shape_view, Stride, Stride_view
#include <span>                   // for span

namespace kaad {

/// @internal
template <bool isMut> struct Tensor_view_impl {
    using value_type = std::conditional_t<isMut, Scalar, const Scalar>;
    using reference = value_type &;
    using const_reference = const value_type &;
    using pointer = value_type *;
    using const_pointer = const value_type *;

    using iterator = value_type *;
    using const_iterator = const value_type *;
    using difference_type = std::ptrdiff_t;
    using size_type = std::size_t;

    Tensor::Shape_view shape;   ///< Dimensions of the tensor.
    Tensor::Stride_view stride; ///< Stride of the tensor (steps needed to move
                                ///< one element in each dimension).
    std::span<value_type> elements; ///< Elements of the tensor.

    /**
     * @brief Default constructor.
     */
    Tensor_view_impl() noexcept;

    /**
     * @brief Constructs a tensor view.
     * @param shape Pointer to the shape array.
     * @param shape Pointer to the stride array.
     * @param rank Length of the shape and stride arrays.
     * @param elements Pointer to the element array.
     * @param len Length of the element array.
     */
    Tensor_view_impl(Tensor::Shape_view shape, Tensor::Stride_view stride,
                     std::span<value_type> elements)
        : shape(shape), stride(stride), elements(elements) {}

    /// @brief Get rank of the tensor.
    /// @return Length of the shape array.
    [[nodiscard]] size_type rank() const { return this->shape.size(); }
};

/**
 * @brief Overloads the stream output operator to print the tensor view.
 *
 * @param stream Output stream.
 * @param view The tensor view to print.
 * @return std::ostream& The updated output stream.
 */
template <bool isMut>
std::ostream &operator<<(std::ostream &stream,
                         const Tensor_view_impl<isMut> &tensor) {
    print_tensor_impl(stream, tensor.shape, tensor.stride, tensor.elements);
    return stream;
}

/// @brief Non-owning immutable view of a tensor.
struct Tensor_view : Tensor_view_impl<false> {
    using Tensor_view_impl<false>::Tensor_view_impl;
};

/// @brief Non-owning mutable view of a tensor.
struct Tensor_view_mut : Tensor_view_impl<true> {
    using Tensor_view_impl<true>::Tensor_view_impl;
};

} // namespace kaad
