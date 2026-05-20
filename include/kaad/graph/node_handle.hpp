#pragma once

#include "kaad/tensor/internal/tensor_types.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <kaad/tensor/tensor_view.hpp>

namespace kaad {

class Graph;
struct INode;

/// @brief Immutable handle class for a INode.
class Node {
    uint32_t idx_;  ///< Index of the node in the Graph.
    Graph *origin_; ///< Pointer to the correct Graph.

    constexpr explicit Node(uint32_t idx, Graph *origin) noexcept
        : idx_(idx), origin_(origin) {}

    /// @return A non-const pointer to the node interface.
    [[nodiscard]] INode *node_mut();

    friend class Graph;

  public:
    /// @return Index of the node in the @ref kaad::Graph.
    [[nodiscard]] constexpr uint32_t idx() const noexcept { return this->idx_; }

    /// @return Pointer to the @ref kaad::Graph.
    [[nodiscard]] constexpr const Graph *origin() const noexcept {
        return this->origin_;
    }

    /// @return A const pointer to the node interface.
    [[nodiscard]] const INode *node() const;

    /// @return The name of the operation performed by the node.
    [[nodiscard]] const char *operation_name() const;

    /// @return The rank of the nodes value and gradient tensors.
    [[nodiscard]] std::size_t rank() const;

    /// @return Number of elements in the value tensor (the gradient tensor has
    /// the same size).
    [[nodiscard]] std::size_t size() const;

    /// @return The shape of the nodes value and gradient tensors.
    [[nodiscard]] ShapeView shape() const;

    /// @return A view of the value tensor.
    [[nodiscard]] TensorView value() const;

    /// @return A view of the gradient tensor.
    [[nodiscard]] TensorView gradient() const;

    /// @return Pointer to the underlying elements.
    /// @throws kaad::LogicError If is_input() is false.
    Scalar *data_mut();

    /// @return True if node is evaluated false otherwise.
    [[nodiscard]] bool is_evaluated() const;

    /// @return True if node is an InputNode false otherwise.
    [[nodiscard]] bool is_input() const;

    friend constexpr auto operator<=>(Node, Node) = default;
    friend std::ostream &operator<<(std::ostream &stream, Node node);
};

} // namespace kaad
