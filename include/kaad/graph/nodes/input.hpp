#pragma once

#include "inode.hpp" // for INode
#include <span>      // for span

namespace kaad {

/**
 * @brief A leaf node that holds a fixed tensor value with no computation.
 * @ingroup nodes
 */
class Node_input : public INode {
  private:
    /// No evaluation needed for fixed value nodes.
    void eval() override {}

    /// No gradient computation needed for fixed value nodes.
    void getGrad() override {}

  public:
    /**
     * @brief Constructs a leaf node holding a constant value.
     * @ingroup nodes
     * @param tensor The tensor value to store.
     * @param label Label string for the node.
     */
    Node_input(std::span<const int> value_shape, const char *label = "")
        : INode(value_shape, true, label) {}

    /**
     * @brief Returns the type of the node as a string.
     * @ingroup nodes
     */
    [[nodiscard]] const char *node_type() const noexcept override {
        return "Node_input";
    }
};

} // namespace kaad
