#pragma once

#include <cstddef>                               // for size_t
#include <cstdint>                               // for uint32_t
#include <iostream>                              // for ostream
#include <kaad/tensor/internal/tensor_types.hpp> // for ShapeView
#include <kaad/tensor/tensor_view.hpp> // for TensorViewConst, TensorViewMut

namespace kaad {

class Graph;
class INode;

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

    /// @return The shape of the nodes value and gradient tensors.
    [[nodiscard]] ShapeView shape() const;

    /// @return An immutable view of the value tensor.
    [[nodiscard]] TensorViewConst value() const;

    /// @throws kaad::LogicError if called on a non-input node.
    /// @return An mutable view of the value tensor.
    [[nodiscard]] TensorViewMut value_mut();

    /// @return An mutable view of the gradient tensor.
    [[nodiscard]] TensorViewConst gradient() const;

    /// @return True if node is evaluated false otherwise.
    [[nodiscard]] bool is_evaluated() const;

    /// @return True if node is an @ref InputNode false otherwise.
    [[nodiscard]] bool is_input() const;

    friend constexpr auto operator<=>(Node, Node) = default;
    friend std::ostream &operator<<(std::ostream &stream, Node node);
};

} // namespace kaad
