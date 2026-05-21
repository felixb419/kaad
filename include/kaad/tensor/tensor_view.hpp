#pragma once

#include "kaad/tensor/internal/iterator_impl.hpp"
#include "kaad/tensor/internal/tensor.hpp"
#include "kaad/tensor/internal/tensor_types.hpp"

#include <cstddef>
#include <iostream>
#include <kaad/enums.hpp>
#include <kaad/scalar.hpp>
#include <span>

namespace kaad {

struct Tensor;

/// @brief A non-owning immutable view of a tensor object.
class TensorView {
  private:
    const Tensor *origin_; ///< Tensor the view is based on.

  public:
    using value_type = const Scalar;
    using reference = value_type &;
    using const_reference = const value_type &;
    using pointer = value_type *;
    using const_pointer = const value_type *;
    using difference_type = std::ptrdiff_t;
    using size_type = std::size_t;

    using iterator = IteratorImpl<IMMUTABLE>;

    /// @brief Default constructor.
    TensorView() = default;

    /**
     * @brief Constructs a tensor view.
     * @param tensor Pointer to the origin of the view.
     */
    TensorView(const Tensor *tensor);

    /// @brief Get shape of the tensor.
    /// @return Read-only span representing the extent of the tensor along every
    /// axis.
    [[nodiscard]] ShapeView shape() const noexcept;

    /// @brief Get strides of the tensor.
    /// @return Read-only span representing the strides array.
    [[nodiscard]] StridesView strides() const noexcept;

    /**
     * @brief Get a std::span representing the elements array.
     * @note If the tensor has been transposed the physical and logical
     * order of elements can differ.
     * @return Read-only span representing the elements.
     */
    [[nodiscard]] std::span<const value_type> elements() const noexcept;

    /// @brief Get rank of the tensor.
    /// @return Length of the shape array.
    [[nodiscard]] size_type rank() const;

    /// @return Extent of tensor along specified axis (shape[@p axis]).
    [[nodiscard]] Extent extent(size_type axis) const noexcept;

    /// @return True if tensor is a scalar false otherwise.
    [[nodiscard]] bool scalar() const noexcept;

    /// @return True if the tensor elements are contiguous in memory, false
    /// otherwise.
    [[nodiscard]] bool is_contiguous() const noexcept;

    /**
     * @brief Returns an iterator to the first logical element.
     * @note Iterates in logical (transposed) order; use @c data() for
     * contiguous raw access.
     * @return Read iterator to the first element.
     */
    [[nodiscard]] iterator begin() const noexcept;

    /**
     * @brief Returns an iterator to one past the last logical element.
     * @note Iterates in logical (transposed) order; use @c data() for
     * contiguous raw access.
     * @return Read iterator to one past the last element.
     */
    [[nodiscard]] iterator end() noexcept;

    /// @brief Get number of elements in the tensor.
    /// @return Length of the element array.
    [[nodiscard]] size_type size() const noexcept;

    /// @return True if tensor has no elements, false otherwise.
    [[nodiscard]] bool empty() const noexcept;
};

/**
 * @brief Overloads the stream output operator to print the tensor view.
 *
 * @param stream Output stream.
 * @param tensor The tensor view to print.
 * @return std::ostream& The updated output stream.
 */
std::ostream &operator<<(std::ostream &stream, const TensorView &tensor);

} // namespace kaad
