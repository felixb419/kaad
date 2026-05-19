#include "kaad/graph/internal/inode.hpp"
#include "kaad/tensor/internal/tensor.hpp"
#include "kaad/tensor/internal/tensor_types.hpp"

#include <cstddef>
#include <kaad/exceptions.hpp>
#include <kaad/graph/graph.hpp>
#include <kaad/graph/node_handle.hpp>
#include <kaad/scalar.hpp>
#include <kaad/tensor/tensor_view.hpp>
#include <ostream>
#include <vector>

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

[[nodiscard]] std::size_t Node::size() const {
    return this->node()->value.size;
}

[[nodiscard]] ShapeView Node::shape() const { return this->node()->shape(); }

[[nodiscard]] TensorView Node::value() const {
    return this->node()->value.view();
}

[[nodiscard]] TensorView Node::gradient() const {
    return this->node()->gradient.view();
}

Scalar *Node::data_mut() {

    if (!this->is_input()) {

        throw LogicError("data_mut may not be called on not input nodes");
    }

    return this->node_mut()->value.data;
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
