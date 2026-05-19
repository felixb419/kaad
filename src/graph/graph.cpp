#include "input_node.hpp"
#include "kaad/graph/internal/inode.hpp"
#include "kaad/tensor/internal/tensor.hpp"
#include "kaad/tensor/internal/tensor_types.hpp"

#include <algorithm>
#include <cstddef>
#include <kaad/exceptions.hpp>
#include <kaad/graph/graph.hpp>
#include <kaad/graph/node_handle.hpp>
#include <kaad/scalar.hpp>
#include <kaad/tensor/tensor_view.hpp>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace kaad {

Node Graph::back_handle() noexcept {
    return Node(this->nodes.size() - 1, this);
}

INode *Graph::get_node(Node node) {
    if (node.origin_ != this) {
        throw ArgumentError("node does not belong to this instance of Graph");
    }
    if (node.idx_ >= this->nodes.size()) {
        throw ArgumentError(std::to_string(node.idx_) +
                            "is not a valid index for this Graph");
    }

    return this->nodes[node.idx_].get();
}

void Graph::allocate() {

    std::size_t total_elements = 0;

    for (auto &node_ptr : this->nodes) {

        total_elements += node_ptr->value.size + node_ptr->gradient.size;
    }

    this->tensor_buff.resize(total_elements);

    Scalar *buffer_iter = this->tensor_buff.data();
    for (auto &node_ptr : this->nodes) {

        node_ptr->value.data = buffer_iter;
        buffer_iter += node_ptr->value.size;

        node_ptr->gradient.data = buffer_iter;
        buffer_iter += node_ptr->gradient.size;
    }
}

void Graph::init() {

    this->allocate();

    for (std::unique_ptr<INode> &node : this->nodes) {
        node->init_params();
    }
}

Node Graph::add_input_node(ShapeView value_shape) {
    this->nodes.push_back(std::make_unique<InputNode>(value_shape));

    return Node(this->nodes.size() - 1, this);
}

void Graph::reset() {
    for (std::unique_ptr<INode> &node : this->nodes) {
        (*node).reset();
    }
}

std::vector<TensorView> Graph::evaluate(std::span<const Node> nodes) {

    std::vector<TensorView> values(nodes.size());

    for (std::size_t i = 0; i < nodes.size(); i++) {
        INode *node_ptr = this->get_node(nodes[i]);
        node_ptr->evaluate();
        values[i] = node_ptr->value.view();
    }

    return values;
}

std::vector<TensorView> Graph::get_gradient(Node output,
                                            std::span<const Node> inputs) {

    INode *output_node = this->get_node(output);
    std::fill_n(output_node->gradient.data, output_node->gradient.size, 1.0);

    output_node->acc_input_gradients();

    std::vector<TensorView> partials(inputs.size());

    for (std::size_t i = 0; i < inputs.size(); i++) {
        partials[i] = this->get_node(inputs[i])->gradient.view();
    }

    return partials;
}

} // namespace kaad
