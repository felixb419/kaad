#pragma once

#include "../../tensor/tensor.hpp" // for Tensor
#include <utility>                 // for std::forward

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
 *
 * @tparam T The scalar type (e.g., float or double).
 */
template <typename T> class INode {
  public:
    virtual const char *node_type() const noexcept = 0;

    INode<T> *A =
        nullptr; ///< Pointer to the first input Node (nullptr if leaf node).

    Tensor<T> value;    ///< Value computed by this node.
    Tensor<T> gradient; ///< Gradient associated with this node.

    bool evaluated = false; ///< Whether this node is currently evaluated.
    bool hasInputs = false; ///< Whether this node depends on any input nodes.

    /**
     * @brief Constructs an unevaluated node with a dependency on an input node.
     *
     * @param A_ptr Pointer to the input node.
     * @param tensor_args Arguments to construct the value tensor.
     */
    template <typename... TensorArgs>
    INode(INode<T> *A_ptr, TensorArgs &&...tensor_args)
        : A(A_ptr), evaluated(false),
          value(std::forward<TensorArgs>(tensor_args)...),
          hasInputs(this->A != nullptr), gradient(value) {
        if (this->hasInputs) {
            std::fill(value.elements_.begin(), value.elements_.end(), 0);
        }
        std::fill(gradient.elements_.begin(), gradient.elements_.end(), 0);
    }

    /// Virtual destructor for polymorphic deletion
    virtual ~INode() = default;

    /**
     * @brief Resets the value and gradient of the node.
     *
     * Clears the value tensor and sets the `evaluated` flag to false if the
     * node has inputs. Clears the gradient tensor in all cases.
     */
    inline void reset() {
        if (hasInputs) {
            std::fill(value.elements_.begin(), value.elements_.end(), 0);
            evaluated = false;
        }
        std::fill(gradient.elements_.begin(), gradient.elements_.end(), 0);
    }

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
