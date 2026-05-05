#pragma once

#include <concepts>
#include <kaad/tensor/tensor.hpp>
#include <kaad/tensor/tensor_view.hpp>
#include <ranges>

namespace kaad {

class INode {
  protected:
    Tensor value_;
    Tensor gradient_;

    INode(ShapeView shape)
        : value_(Tensor::zeros(shape)), gradient_(Tensor::zeros(shape)) {}

    INode(std::pair<ShapeView, StridesView> shape_pair)
        : value_(shape_pair.first, shape_pair.second),
          gradient_(shape_pair.first, shape_pair.second) {}

  public:
    virtual ~INode() = default;

    [[nodiscard]] virtual const char *operation_name() const noexcept = 0;

    [[nodiscard]] std::size_t rank() const noexcept {
        return this->value_.rank();
    }

    [[nodiscard]] ShapeView shape() const { return this->value_.shape(); }

    [[nodiscard]] TensorViewConst value() const { return this->value_.view(); }

    [[nodiscard]] TensorViewMut value_mut() { return this->value_.view_mut(); }

    [[nodiscard]] TensorViewConst gradient() const {
        return this->gradient_.view();
    }

    [[nodiscard]] TensorViewMut gradient_mut() {
        return this->gradient_.view_mut();
    }

    [[nodiscard]] virtual bool is_evaluated() const noexcept = 0;

    [[nodiscard]] virtual bool is_input() const noexcept = 0;

    /// Evaluates the node's value Tensor.
    virtual void evaluate() = 0;

    /// Accumulate the nodes's gradient Tensor.
    virtual void acc_input_gradients() = 0;

    /// Reset evaluated_ flag, fill value and gradient tensor with 0.
    virtual void reset() = 0;
};

} // namespace kaad
