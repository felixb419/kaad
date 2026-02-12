#pragma once

#include <cstdint>  // for uint32_t
#include <iostream> // for std::ostream

namespace kaad {
class Computation_graph;

/**
 * @brief Immutable handle class for a INode.
 */
class Node_handle {
    uint32_t idx_;              ///< Index of the node in the Computation_graph.
    Computation_graph *origin_; ///< Pointer to the correct Computation_graph.

    /**
     * @brief private constructor
     * @param idx Index of the node in the Computation_graph.
     * @param origin Pointer to the Computation_graph.
     */
    constexpr explicit Node_handle(uint32_t idx,
                                   Computation_graph *origin) noexcept
        : idx_(idx), origin_(origin) {}

    friend class Computation_graph;

  public:
    /**
     * @brief Get the index.
     * @return Copy of the idx_ member.
     */
    constexpr uint32_t idx() const noexcept { return this->idx_; }

    /**
     * @brief Get the origin of the Node.
     * @return Pointer to the Computation_graph which contains the node.
     */
    constexpr const Computation_graph *origin() const noexcept {
        return this->origin_;
    }

    friend constexpr auto operator<=>(Node_handle, Node_handle) = default;
    friend std::ostream &operator<<(std::ostream &os, Node_handle node);
};

} // namespace kaad
