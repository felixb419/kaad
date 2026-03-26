#pragma once

#include <algorithm>                    // for reverse_copy
#include <cstddef>                      // for size_t
#include <iostream>                     // for ostream, ptrdiff_t
#include <kaad/mutability_enum.hpp>     // for MUTABILITY
#include <kaad/scalar.hpp>              // for Scalar
#include <kaad/tensor/tensor_types.hpp> // for Shape, ShapeView, Stride, StrideView
#include <span>                         // for span

namespace kaad {

/// @brief A non-owning view of a tensor object, mutability is dependant on the
/// constness of @p T.
/// @tparam T Element type.
template <MUTABILITY M> struct TensorView {
    static constexpr bool IS_MUT = (M == MUTABLE);

    using value_type = std::conditional_t<IS_MUT, Scalar, const Scalar>;
    using reference = value_type &;
    using const_reference = const value_type &;
    using pointer = value_type *;
    using const_pointer = const value_type *;

    using iterator = value_type *;
    using const_iterator = const value_type *;
    using difference_type = std::ptrdiff_t;
    using size_type = std::size_t;

    ShapeView shape;   ///< Dimensions of the tensor.
    StrideView stride; ///< Stride of the tensor (steps needed to move
                       ///< one element in each dimension).
    std::span<value_type> elements; ///< Elements of the tensor.

    /**
     * @brief Default constructor.
     */
    TensorView() noexcept;

    /**
     * @brief Constructs a tensor view.
     * @param shape Pointer to the shape array.
     * @param shape Pointer to the stride array.
     * @param rank Length of the shape and stride arrays.
     * @param elements Pointer to the element array.
     * @param len Length of the element array.
     */
    TensorView(ShapeView shape, StrideView stride,
               std::span<value_type> elements)
        : shape(shape), stride(stride), elements(elements) {}

    /// @brief Get rank of the tensor.
    /// @return Length of the shape array.
    [[nodiscard]] size_type rank() const { return this->shape.size(); }

    /**
     * @brief Get a transposed copy of the view.
     * @param shape_buff The new shape will be stored in this.
     * @param stride_buff The new strides will be stored in this.
     * @note The memory of @p shape_buff has to be manually freed.
     * @return View with transposed shape and stride.
     */
    TensorView<M> transpose(Shape &shape_buff, Stride &stride_buff) const {

        std::size_t rank = this->rank();

        shape_buff.resize(rank);
        std::ranges::reverse_copy(this->shape, shape_buff.begin());

        stride_buff.resize(rank);
        std::ranges::reverse_copy(this->stride, stride_buff.begin());

        TensorView<M> out = *this;

        out.shape = ShapeView(shape_buff);
        out.stride = ShapeView(stride_buff);

        return out;
    }

    /**
     * @brief Get a copy of the view with the lowest 2 dimensions transposed.
     * @param shape_buff The new shape will be stored in this.
     * @param stride_buff The new strides will be stored in this.
     * @note The memory of @p shape_buff has to be manually freed.
     * @return View with transposed shape and stride.
     */
    TensorView<M> transpose_2d(Shape &shape_buff, Stride &stride_buff) {

        std::size_t rank = this->rank();

        shape_buff = Shape(this->shape);
        std::swap(shape_buff[rank - 1], shape_buff[rank - 2]);

        stride_buff = Stride(this->stride);

        std::swap(stride_buff[rank - 1], stride_buff[rank - 2]);

        TensorView<M> out = *this;

        out.shape = ShapeView(shape_buff);
        out.stride = ShapeView(stride_buff);

        return out;
    }
};

/**
 * @brief Overloads the stream output operator to print the tensor view.
 *
 * @param stream Output stream.
 * @param view The tensor view to print.
 * @return std::ostream& The updated output stream.
 */
template <MUTABILITY M>
std::ostream &operator<<(std::ostream &stream, const TensorView<M> &tensor) {
    print_tensor_impl(stream, tensor.shape, tensor.stride, tensor.elements);
    return stream;
}

/// @copybrief TensorView
/// Provides immutable access to the elements.
using TensorViewConst = TensorView<IMMUTABLE>;

/// @copybrief TensorView
/// Provides mutable access to the elements.
using TensorViewMut = TensorView<MUTABLE>;

} // namespace kaad
