#pragma once

#include <kaad/tensor/tensor.hpp>       // for Tensor
#include <kaad/tensor/tensor_types.hpp> // for ShapeView, StrideView

namespace kaad {

/**
 * @defgroup nodes Nodes that can be added to a @ref Graph.
 */

/**
 * @brief Abstract base class representing a node in a computation graph.
 * @ingroup nodes
 *
 * Each node holds a value tensor and a corresponding gradient tensor.
 * Nodes can be evaluated to compute their output, and gradients can be
 * backpropagated through them. Derived classes must implement the `eval` and
 * `get_grad` methods. Traversal-related members (e.g., strides, offsets) are
 * not initialized in the node constructors but are instead set externally by
 * the corresponding helper functions in the Strides namespace.
 */
class INode {
  protected:
    Tensor value_;    ///< Value computed by this node.
    Tensor gradient_; ///< Gradient associated with this node.

    const char *label_; ///< User set name of the node.

    bool evaluated_ = false;     ///< Whether this node is currently evaluated.
    bool is_input_node_ = false; ///< Whether this node is an input node.

    /**
     * @brief Initializes the first input, flags, value and gradient tensors for
     * a node.
     * @ingroup nodes
     * @param value_shape Output/gradient shape
     * @param is_input_node Flag indicating wheter it is an input node.
     * @param label Label string for the node.
     * @param value_strides Stride array of the value tensor (can be omitted if
     * tensor is contigous).
     */
    INode(ShapeView value_shape, bool is_input_node, const char *label = "",
          StridesView value_strides = {});

  public:
    /// Virtual destructor for polymorphic deletion
    virtual ~INode() = default;

    /// @return Type of the node as a string.
    [[nodiscard]] virtual const char *node_type() const noexcept = 0;

    /**
     * @return Lable of the node as a string.
     */
    const char *label() { return this->label_; }

    /// @return Reference to the value tensor.
    Tensor &value() noexcept;

    /// @return Const reference to the value tensor.
    [[nodiscard]] const Tensor &value() const noexcept;

    /// @return Reference to the value tensor.
    Tensor &gradient() noexcept;

    /// @return Const reference to the value tensor.
    [[nodiscard]] const Tensor &gradient() const noexcept;

    /// @return True if the node has already been evaluated, false otherwise.
    [[nodiscard]] bool evaluated() const noexcept;

    /// @return True if the node is an input node, false otherwise.
    [[nodiscard]] bool is_input() const noexcept;

    /**
     * @brief Resets the value and gradient of the node.
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
    inline virtual void get_grad() = 0;

    friend class Computation_grap;
};

} // namespace kaad
