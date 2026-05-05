#pragma once

#include <cstdint>
#include <kaad/enums.hpp>
#include <kaad/graph/inode.hpp>
#include <kaad/graph/operation_concept.hpp>
#include <kaad/tensor/tensor.hpp>
#include <kaad/tensor/tensor_view.hpp>

namespace kaad {

template <Operation operation> class OperatorNode : public INode {
  private:
    std::array<INode *, operation::ARITY> inputs_;

    /// Parameters needed for forward pass
    operation::ForwardParams fp_;

    /// Parameters needed for backward pass
    operation::BackwardParams bp_;

    operation::Dispatch functions_;

    bool is_evaluated_ = false;

  public:
    template <typename... ExtraParams>
    OperatorNode(const std::array<INode *, operation::ARITY> &inputs,
                 ExtraParams... extra)
        : INode(operation::make_res_shape(inputs, extra...)), inputs_(inputs),
          fp_(inputs, this, extra...), bp_(inputs, this, extra...),
          functions_(operation::dispatch(inputs, this, extra...)) {}

    [[nodiscard]] const char *operation_name() const noexcept override {
        return operation::OPERATION_NAME;
    }

    [[nodiscard]] bool is_evaluated() const noexcept override {
        return this->is_evaluated_;
    }

    [[nodiscard]] bool is_input() const noexcept override { return false; }

    void evaluate() override {
        if (!this->is_evaluated_) {

            for (INode *ptr : this->inputs_) {
                ptr->evaluate();
            }

            functions_.forward(this->fp_);

            this->is_evaluated_ = true;
        }
    }

    /// Accumulate the gradients of the input nodes.
    void acc_input_gradients() override {

        functions_.backward(this->bp_);

        for (INode *ptr : this->inputs_) {

            if (!ptr->is_input()) {
                ptr->acc_input_gradients();
            }
        }
    }

    void reset() override {
        this->is_evaluated_ = false;

        std::ranges::fill(this->value_.elements(), Tensor::value_type{});
        std::ranges::fill(this->gradient_.elements(), Tensor::value_type{});
    }
};

} // namespace kaad
