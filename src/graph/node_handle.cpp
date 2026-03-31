#include <kaad/graph/node_handle.hpp>

#include <cassert>                    // for assert
#include <kaad/exceptions.hpp>        // for ArgumentError
#include <kaad/graph/graph.hpp>       // for Graph
#include <kaad/graph/nodes/inode.hpp> // for INode
#include <kaad/scalar.hpp>            // for Scalar
#include <kaad/tensor/tensor.hpp>     // for operator<<, Tensor
#include <memory>                     // for unique_ptr
#include <vector>                     // for vector

namespace kaad {

[[nodiscard]] const INode *Node::node() const {
    if (this->idx() >= this->origin_->nodes.size()) {
        throw ArgumentError("idx_ of this handle is invalid");
    }

    return this->origin_->nodes[this->idx()].get();
}

[[nodiscard]] ShapeView Node::shape() const { return this->value().shape(); }

std::span<Scalar> Node::value_elements() {
    assert(this->node()->is_input());

    Tensor &value = this->origin_->get_node(*this)->value();
    return {value.data(), value.size()};
}

[[nodiscard]] const Tensor &Node::value() const {
    return this->node()->value();
}

[[nodiscard]] const Tensor &Node::gradient() const {
    return this->node()->gradient();
}

std::ostream &operator<<(std::ostream &stream, Node node) {
    stream << node.node()->node_type() << " at idx " << node.idx()
           << " of Graph at " << node.origin() << "\n"
           << "value:\n"
           << node.value() << "\ngradient:\n"
           << node.gradient();
    return stream;
}

} // namespace kaad
