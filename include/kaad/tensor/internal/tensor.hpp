#pragma once

#include "kaad/tensor/internal/iterator_impl.hpp"
#include "kaad/tensor/internal/tensor_types.hpp"

#include <cstddef>
#include <iostream>
#include <kaad/enums.hpp>
#include <kaad/scalar.hpp>

namespace kaad {

class TensorView;

/// @brief A class representing a multi-dimensional tensor.
/// @c elements must be manually set after construction.
struct Tensor {
    static constexpr const std::size_t MAX_RANK = Shape::MAX_SIZE;

    Shape shape;     ///< Array containing the size along each axis.
    Strides strides; ///< Array containing the strides (steps needed to move one
                     ///< element along each axis).

    Scalar *data;     ///< Pointer to the first element.
    std::size_t size; ///< Number of elements.

    /// @return Strides array based on @p shape.
    static Strides compute_strides(ShapeView shape) noexcept;

    /// @return Number of tensor elements based on @p shape.
    static std::size_t compute_size(ShapeView shape);

    Tensor() = default;

    explicit Tensor(ShapeView shape);

    explicit Tensor(ShapeView shape, StridesView strides);

    [[nodiscard]] TensorView view() const noexcept;

    [[nodiscard]] std::size_t rank() const noexcept;

    [[nodiscard]] bool scalar() const noexcept;

    [[nodiscard]] Tensor transpose(StaticVector<std::size_t> perm = {}) const;

    [[nodiscard]] Tensor transpose_2d() const;

    friend class Graph;
    friend struct INode;
};

std::ostream &operator<<(std::ostream &stream, const Tensor &tensor);

} // namespace kaad
