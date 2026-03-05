#pragma once

#include "../../tensor/tensor.hpp" // for Tensor
#include <span>                    // for span

namespace kaad {

/**
 * @defgroup nodes Nodes that can be added to a @ref Computation_graph.
 */

/**
 * @brief Abstract base class representing a node in a computation graph.
 * @ingroup nodes
 *
 * Each node holds a value tensor and a corresponding gradient tensor.
 * Nodes can be evaluated to compute their output, and gradients can be
 * backpropagated through them. Derived classes must implement the `eval` and
 * `getGrad` methods. Traversal-related members (e.g., strides, offsets) are not
 * initialized in the node constructors but are instead set externally by the
 * corresponding helper functions in the Strides namespace.
 */
class INode {
  protected:
    Tensor value_;    ///< Value computed by this node.
    Tensor gradient_; ///< Gradient associated with this node.

    bool evaluated_ = false; ///< Whether this node is currently evaluated.
    bool hasInputs_ = false; ///< Whether this node depends on any input nodes.

    /**
     * @brief Initializes the first input, flags, value and gradient tensors for
     * a node.
     * @ingroup nodes
     * @param value_shape Shape of the value and gradient tensor.
     * @param is_input_node Flag indicating wheter it is an input node.
     * @param value_stride Stride array of the value tensor (can be omitted if
     * value is not transposed).
     */
    INode(std::span<const int> value_shape, bool is_input_node,
          std::span<const int> value_stride = {});

  public:
    /// Virtual destructor for polymorphic deletion
    virtual ~INode() = default;

    /**
     * @brief Returns the type of the node as a string.
     * @ingroup nodes
     */
    virtual const char *node_type() const noexcept = 0;

    /**
     * @brief Get a refernce to the value tensor.
     * @ingroup nodes
     * @return Reference to the value tensor.
     */
    Tensor &value() noexcept;

    /**
     * @brief Get a refernce to the value tensor.
     * @ingroup nodes
     * @return Immutable reference to the value tensor.
     */
    const Tensor &value() const noexcept;

    /**
     * @brief Get a refernce to the gradient tensor.
     * @ingroup nodes
     * @return Immutable reference to the gradient tensor.
     */
    Tensor &gradient() noexcept;

    /**
     * @brief Get a refernce to the gradient tensor.
     * @ingroup nodes
     * @return Reference to the gradient tensor.
     */
    const Tensor &gradient() const noexcept;

    /**
     * @brief Tells wether the node has been evaluated.
     * @ingroup nodes
     * @return True if the node was already evaluated, False otherwise.
     */
    bool evaluated() const noexcept;

    /**
     * @brief Tells wether the node has inputs.
     * @ingroup nodes
     * @return True if the node has inputs, False otherwise.
     */
    bool hasInputs() const noexcept;

    /**
     * @brief Resets the value and gradient of the node.
     * @ingroup nodes
     *
     * Clears the value tensor and sets the `evaluated` flag to false if the
     * node has inputs. Clears the gradient tensor in all cases.
     */
    void reset();

    /**
     * @brief Evaluates the node's value. Must be implemented by derived
     * @ingroup nodes
     * classes.
     */
    inline virtual void eval() = 0;
    /**
     * @brief Computes the gradient for this node. Must be implemented by
     * @ingroup nodes
     * derived classes.
     */
    inline virtual void getGrad() = 0;

    friend class Computation_grap;
};

} // namespace kaad
