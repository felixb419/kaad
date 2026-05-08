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

    [[nodiscard]] INode *node_mut();

    friend class Graph;

  public:
    [[nodiscard]] constexpr uint32_t idx() const noexcept { return this->idx_; }

    [[nodiscard]] constexpr const Graph *origin() const noexcept {
        return this->origin_;
    }

    [[nodiscard]] const INode *node() const;

    [[nodiscard]] const char *operation_name() const;

    [[nodiscard]] std::size_t rank() const;

    [[nodiscard]] ShapeView shape() const;

    [[nodiscard]] TensorViewConst value() const;

    [[nodiscard]] TensorViewMut value_mut();

    [[nodiscard]] TensorViewConst gradient() const;

    [[nodiscard]] bool is_evaluated() const;

    [[nodiscard]] bool is_input() const;

    friend constexpr auto operator<=>(Node, Node) = default;
    friend std::ostream &operator<<(std::ostream &stream, Node node);
};

} // namespace kaad
