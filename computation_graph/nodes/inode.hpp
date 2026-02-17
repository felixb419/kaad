#pragma once

#include "../../tensor/tensor.hpp" // for Tensor

namespace kaad {

/**
 * @brief Abstract base class representing a node in a computation graph.
 *
 * Each node holds a value tensor and a corresponding gradient tensor.
 * Nodes can be evaluated to compute their output, and gradients can be
 * backpropagated through them. Derived classes must implement the `eval` and
 * `getGrad` methods. Traversal-related members (e.g., strides, offsets) are not
 * initialized in the node constructors but are instead set externally by the
 * corresponding helper functions in the Strides namespace.
 */
class INode {
  public:
    /**
     * @brief Returns the type of the node as a string.
     */
    virtual const char *node_type() const noexcept = 0;

    Tensor value;    ///< Value computed by this node.
    Tensor gradient; ///< Gradient associated with this node.

    bool evaluated = false; ///< Whether this node is currently evaluated.
    bool hasInputs = false; ///< Whether this node depends on any input nodes.

    /**
     * @brief Initializes the first input, flags, value and gradient tensors for
     * a node.
     *
     * @param A_ptr Pointer to the input node.
     * @param value_shape Shape of the value and gradient tensor.
     */
    INode(std::span<const int> value_shape, bool is_input_node);

    /**
     * @brief Initializes the first input, flags, value and gradient tensors for
     * a node.
     *
     * @param A_ptr Pointer to the input node.
     * @param value_shape Shape of the value and gradient tensor.
     * @param value_stride Stride array of the value and gradient tensor.
     */
    INode(std::span<const int> value_shape, std::span<const int> value_stride,
          bool is_input_node);

    /// Virtual destructor for polymorphic deletion
    virtual ~INode() = default;

    /**
     * @brief Resets the value and gradient of the node.
     *
     * Clears the value tensor and sets the `evaluated` flag to false if the
     * node has inputs. Clears the gradient tensor in all cases.
     */
    void reset();

    /**
     * @brief Evaluates the node's value. Must be implemented by derived
     * classes.
     */
    inline virtual void eval() = 0;
    /**
     * @brief Computes the gradient for this node. Must be implemented by
     * derived classes.
     */
    inline virtual void getGrad() = 0;
};

} // namespace kaad
