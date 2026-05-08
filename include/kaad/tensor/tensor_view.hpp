#pragma once

#include <algorithm>                     // for reverse_copy
#include <cstddef>                       // for size_t
#include <iostream>                      // for ostream, ptrdiff_t
#include <kaad/enums.hpp>                // for MUTABILITY
#include <kaad/scalar.hpp>               // for Scalar
#include <kaad/tensor/iterator_impl.hpp> // for IteratorImpl
#include <kaad/tensor/print_tensor.hpp>  // for print_tensor_impl
#include <kaad/tensor/tensor_types.hpp> // for Shape, ShapeView, Strides, StridesView
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
    using difference_type = std::ptrdiff_t;
    using size_type = std::size_t;

    using iterator = std::conditional_t<IS_MUT, IteratorImpl<MUTABLE>,
                                        IteratorImpl<IMMUTABLE>>;

    ShapeView shape;     ///< Dimensions of the tensor.
    StridesView strides; ///< Strides of the tensor (steps needed to move
                         ///< one element along each axis).
    std::span<value_type> elements; ///< Elements of the tensor.

    /**
     * @brief Default constructor.
     */
    TensorView() = default;

    /**
     * @brief Constructs a tensor view.
     * @param shape Pointer to the shape array.
     * @param shape Pointer to the strides array.
     * @param rank Length of the shape and stride arrays.
     * @param elements Pointer to the element array.
     * @param len Length of the element array.
     */
    TensorView(ShapeView shape, StridesView strides,
               std::span<value_type> elements)
        : shape(shape), strides(strides), elements(elements) {}

    /// @copydoc Tensor::rank
    [[nodiscard]] size_type rank() const { return this->shape.size(); }

    /// @copydoc Tensor::scalar
    [[nodiscard]] bool scalar() const noexcept { return this->rank() == 0; }

    /// @copydoc Tensor::begin
    [[nodiscard]] iterator begin() const noexcept {
        return {this->shape, this->strides, this->elements, false};
    }

    /// @copydoc Tensor::end
    [[nodiscard]] iterator end() noexcept {
        return {this->shape, this->strides, this->elements, true};
    }

    /// @copydoc Tensor::size
    [[nodiscard]] size_type size() const noexcept {
        return this->elements.size();
    }

    /// @copydoc Tensor::empty
    [[nodiscard]] bool empty() const noexcept { return this->elements.empty(); }

    /// @copydoc Tensor::data
    [[nodiscard]] pointer data() noexcept { return this->elements.data(); }

    /// @copydoc Tensor::data
    [[nodiscard]] const_pointer data() const noexcept {
        return this->elements.data();
    }

    /**
     * @brief Get a transposed copy of the view.
     * @param shape_buff The new shape will be stored in this.
     * @param strides_buff The new strides will be stored in this.
     * @note The memory of @p shape_buff has to be manually freed.
     * @return View with transposed shape and strides.
     */
    TensorView<M> transpose(Shape &shape_buff, Strides &strides_buff) const {

        std::size_t rank = this->rank();

        shape_buff.resize(rank);
        std::ranges::reverse_copy(this->shape, shape_buff.begin());

        strides_buff.resize(rank);
        std::ranges::reverse_copy(this->strides, strides_buff.begin());

        TensorView<M> out = *this;

        out.shape = ShapeView(shape_buff);
        out.strides = ShapeView(strides_buff);

        return out;
    }

    /**
     * @brief Get a copy of the view transposed according to @p perm.
     * @pre @p perm has size equal to @c rank() and doesnt contain
     * duplicates values or values >= @c rank().
     * @param shape_buff The new shape will be stored in this.
     * @param strides_buff The new strides will be stored in this.
     * @param perm Permutation for transposition.
     * @note The memory of @p shape_buff has to be manually freed.
     * @return View with transposed shape and strides.
     */
    TensorView<M> transpose(Shape &shape_buff, Strides &strides_buff,
                            std::span<const std::size_t> perm) const {

        std::size_t rank = this->rank();

        shape_buff.resize(rank);
        strides_buff.resize(rank);

        for (std::size_t i = 0; i < this->rank(); i++) {
            shape_buff[i] = this->shape[perm[i]];
            strides_buff[i] = this->strides[perm[i]];
        }

        TensorView<M> out = *this;

        out.shape = ShapeView(shape_buff);
        out.strides = ShapeView(strides_buff);

        return out;
    }

    /**
     * @brief Get a copy of the view with the last 2 axes transposed.
     * @param shape_buff The new shape will be stored in this.
     * @param strides_buff The new strides will be stored in this.
     * @note The memory of @p shape_buff has to be manually freed.
     * @return View with transposed shape and strides.
     */
    TensorView<M> transpose_2d(Shape &shape_buff, Strides &strides_buff) {

        shape_buff = Shape(this->shape);
        std::swap(shape_buff.from_back(0), shape_buff.from_back(1));

        strides_buff = Strides(this->strides);

        std::swap(strides_buff.from_back(0), strides_buff.from_back(1));

        TensorView<M> out = *this;

        out.shape = ShapeView(shape_buff);
        out.strides = ShapeView(strides_buff);

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
    print_tensor_impl(stream, tensor.shape, tensor.strides, tensor.elements);
    return stream;
}

/// @copybrief TensorView
/// Provides immutable access to the elements.
using TensorViewConst = TensorView<IMMUTABLE>;

/// @copybrief TensorView
/// Provides mutable access to the elements.
using TensorViewMut = TensorView<MUTABLE>;

} // namespace kaad
