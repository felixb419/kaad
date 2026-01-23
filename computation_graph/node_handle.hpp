#pragma once

#include <cstdint>  // for uint32_t
#include <iostream> // for std::ostream

namespace kaad {
template <typename T> class Tensor;
template <typename T> class INode;
template <typename T> class Computation_graph;

/**
 * @brief Immutable handle class for a INode<T>.
 * @tparam T The scalar type (e.g., float or double).
 */
template <typename T> class Node_handle {
    uint32_t idx_; ///< Index of the node in the Computation_graph<T>.
    Computation_graph<T>
        *origin_; ///< Pointer to the correct Computation_graph<T>.

    /**
     * @brief private constructor
     * @param idx Index of the node in the Computation_graph<T>.
     * @param origin Pointer to the Computation_graph<T>.
     */
    constexpr explicit Node_handle(uint32_t idx, Computation_graph<T> *origin)
        : idx_(idx), origin_(origin) {}

    friend class Computation_graph<T>;

  public:
    constexpr const char *node_type() {
        return this->origin_->get_node(*this)->node_type();
    }

    /**
     * @brief Get the index.
     * @return Copy of the idx_ member.
     */
    constexpr uint32_t idx() const noexcept { return this->idx_; }

    /**
     * @brief Get the origin of the Node.
     * @return Pointer to the Computation_graph<T> which contains the node.
     */
    constexpr const Computation_graph<T> *origin() { return this->origin_; }

    /**
     * @brief Get the node.
     * @return Const INode<T> Pointer to the node in the graph.
     */
    constexpr const INode<T> *get() const noexcept {
        return origin_->get_node(*this);
    }

    /**
     * @brief Get the value tensor of the node.
     * @return Immutable reference to the value tensor.
     */
    constexpr const Tensor<T> &value() {
        return this->origin_->nodes[this->idx_].get()->value;
    }

    /**
     * @brief Get the gradient tensor of the node.
     * @return Immutable reference to the gradient tensor.
     */
    constexpr const Tensor<T> &gradient() {
        return this->origin_->nodes[this->idx_].get()->value;
    }

    friend constexpr auto operator<=>(Node_handle, Node_handle) = default;
};

template <typename T>
std::ostream &operator<<(std::ostream &os, Node_handle<T> node) {
    os << node.node_type() << " at idx " << node.idx()
       << " of Computation_graph at " << node.origin() << "\n"
       << "value:\n"
       << node.get()->value << "\ngradient:\n"
       << node.get()->gradient;
    return os;
}

} // namespace kaad
