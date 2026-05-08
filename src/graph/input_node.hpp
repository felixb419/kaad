#pragma once

#include <kaad/graph/internal/inode.hpp>

namespace kaad {

class InputNode : public INode {
  public:
    InputNode(ShapeView shape) : INode(shape) {}

    [[nodiscard]] const char *operation_name() const noexcept override {
        return "input";
    }

    [[nodiscard]] bool is_evaluated() const noexcept override { return true; }

    [[nodiscard]] bool is_input() const noexcept override { return true; }

    void evaluate() noexcept override {}

    void acc_input_gradients() noexcept override {}

    void reset() override {
        std::ranges::fill(this->gradient_.elements(), Tensor::value_type{});
    }
};

} // namespace kaad
