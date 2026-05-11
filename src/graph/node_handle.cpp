#include <cstddef>
#include <kaad/graph/node_handle.hpp>

#include <kaad/exceptions.hpp>                   // for ArgumentError
#include <kaad/graph/graph.hpp>                  // for Graph
#include <kaad/graph/internal/inode.hpp>         // for INode
#include <kaad/tensor/internal/tensor_types.hpp> // for ShapeView
#include <kaad/tensor/tensor.hpp>      // for TensorViewConst, TensorViewMut
#include <kaad/tensor/tensor_view.hpp> // for TensorView, operator<<
#include <ostream>
#include <vector> // for vector

namespace kaad {

[[nodiscard]] INode *Node::node_mut() {
    return const_cast<INode *>(this->node());
}

[[nodiscard]] const INode *Node::node() const {
    if (this->idx() >= this->origin_->nodes.size()) {
        throw ArgumentError("idx_ of this handle is invalid");
    }

    return this->origin_->nodes[this->idx()].get();
}

[[nodiscard]] const char *Node::operation_name() const {
    return this->node()->operation_name();
}

[[nodiscard]] std::size_t Node::rank() const { return this->node()->rank(); }

[[nodiscard]] ShapeView Node::shape() const { return this->node()->shape(); }

[[nodiscard]] TensorViewConst Node::value() const {
    return this->node()->value();
}

[[nodiscard]] TensorViewMut Node::value_mut() {
    if (this->idx() >= this->origin_->nodes.size()) {
        throw ArgumentError("idx_ of this handle is invalid");
    }

    return this->origin_->nodes[this->idx()].get()->value_mut();
}

[[nodiscard]] TensorViewConst Node::gradient() const {
    return this->node()->gradient();
}

[[nodiscard]] bool Node::is_evaluated() const {
    return this->node()->is_evaluated();
}

[[nodiscard]] bool Node::is_input() const { return this->node()->is_input(); }

std::ostream &operator<<(std::ostream &stream, Node node) {
    stream << node.operation_name() << " at idx " << node.idx()
           << " of Graph at " << node.origin() << "\n"
           << "value:\n"
           << node.value() << "\ngradient:\n"
           << node.gradient();
    return stream;
}

} // namespace kaad
