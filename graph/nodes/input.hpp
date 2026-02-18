#pragma once

#include "../../tensor/tensor.hpp" // for Tensor
#include "inode.hpp"               // for INode

namespace kaad {

/**
 * @brief A leaf node that holds a fixed tensor value with no computation.
 */
class Node_input : public INode {
  public:
    /**
     * @brief Returns the type of the node as a string.
     */
    const char *node_type() const noexcept override { return "Node_input"; }

    /**
     * @brief Constructs a leaf node holding a constant value.
     * @param tensor The tensor value to store.
     */
    Node_input(std::span<const int> value_shape) : INode(value_shape, true) {}

    /// No evaluation needed for fixed value nodes.
    void eval() override {}

    /// No gradient computation needed for fixed value nodes.
    void getGrad() override {}
};

} // namespace kaad
