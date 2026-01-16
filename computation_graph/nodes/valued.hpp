#pragma once

#include "../../tensor/tensor.hpp" // for Tensor
#include "inode.hpp"               // for INode

namespace kaad {

/**
 * @brief A leaf node that holds a fixed tensor value with no computation.
 * @tparam T The scalar type.
 */
template <typename T> struct Node_valued : INode<T> {

    /**
     * @brief Constructs a leaf node holding a constant value.
     * @param tensor The tensor value to store.
     */
    Node_valued(Tensor<T> &&tensor) : INode<T>(std::move(tensor)) {}

    /// No evaluation needed for fixed value nodes.
    void eval() override {}
    /// No gradient computation needed for fixed value nodes.
    void getGrad() override {}
};

} // namespace kaad
