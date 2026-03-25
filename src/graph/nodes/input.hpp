#pragma once

#include <kaad/graph/nodes/inode.hpp> // for INode
#include <span>                       // for span

namespace kaad {

/**
 * @brief A input node for a @ref kaad::Graph
 * @ingroup nodes
 * @internal
 */
class NodeInput : public INode {
  private:
    /// No evaluation needed for fixed value nodes.
    void eval() override {}

    /// No gradient computation needed for fixed value nodes.
    void get_grad() override {}

  public:
    /**
     * @brief Construct input node.
     * @param value_shape Output/gradient shape
     * @param label Label string.
     */
    NodeInput(ShapeView value_shape, const char *label = "")
        : INode(value_shape, true, label) {}

    /// @return Type of the node as a string.
    [[nodiscard]] const char *node_type() const noexcept override {
        return "NodeInput";
    }
};

} // namespace kaad
