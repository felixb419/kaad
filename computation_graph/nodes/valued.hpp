#pragma once

#include "../../tensor/tensor.hpp" // for Tensor
#include "inode.hpp"               // for INode

namespace kaad {

/**
 * @brief A leaf node that holds a fixed tensor value with no computation.
 * @tparam T The scalar type.
 */
template <typename T> class Node_valued : public INode<T> {
  public:
    /**
     * @brief Constructs a leaf node holding a constant value.
     * @param tensor The tensor value to store.
     */
    template <typename... TensorArgs>
    Node_valued(TensorArgs &&...tensor_args)
        : INode<T>(nullptr, tensor_args...) {}

    /// No evaluation needed for fixed value nodes.
    void eval() override {}
    /// No gradient computation needed for fixed value nodes.
    void getGrad() override {}
};

} // namespace kaad
