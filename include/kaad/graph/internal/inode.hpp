#pragma once

#include <concepts>
#include <kaad/tensor/tensor.hpp>
#include <kaad/tensor/tensor_view.hpp>
#include <ranges>

namespace kaad {

struct INode {
    Tensor value;
    Tensor gradient;

    INode(std::pair<ShapeView, StridesView> shape_pair)
        : value(shape_pair.first, shape_pair.second),
          gradient(shape_pair.first, shape_pair.second) {

        if (this->value.size == 0) {
            throw CapacityError(
                "a tensor with no elements is not valid for a node, "
                "value.size()=" +
                std::to_string(this->value.size) +
                ", value.shape()=" + to_string(this->value.shape));
        }
    }

    INode(ShapeView shape)
        : INode(std::pair{shape, Tensor::compute_strides(shape)}) {}

    virtual ~INode() = default;

    [[nodiscard]] virtual const char *operation_name() const noexcept = 0;

    [[nodiscard]] std::size_t rank() const noexcept {
        return this->value.rank();
    }

    [[nodiscard]] ShapeView shape() const { return this->value.shape; }

    [[nodiscard]] virtual bool is_evaluated() const noexcept = 0;

    [[nodiscard]] virtual bool is_input() const noexcept = 0;

    /// Reset evaluated_ flag, fill value and gradient tensor with 0.
    virtual void reset() = 0;

    /// Evaluates the node's value Tensor.
    virtual void evaluate() = 0;

    /// Accumulate the nodes's gradient Tensor.
    virtual void acc_input_gradients() = 0;
};

} // namespace kaad
