#pragma once

#include "../operations/operation_concept.hpp"
#include "kaad/graph/internal/inode.hpp"
#include "kaad/tensor/internal/tensor.hpp"

#include <cstdint>
#include <kaad/enums.hpp>
#include <kaad/tensor/tensor_view.hpp>

namespace kaad {

/// Trait to check if an operation class needs a metadata.
template <typename T, typename = void> struct HasMetadata : std::false_type {};

/// @copydoc HasMetadata
template <Operation operation>
struct HasMetadata<operation, std::void_t<typename operation::Metadata>>
    : std::true_type {};

/// Empty struct if operation::Metadata is not present
template <Operation operation, typename = void> struct MetadataBase {};

/// Struct containing Metadata if operation::Metadata is present.
template <Operation operation>
struct MetadataBase<operation, std::void_t<typename operation::Metadata>> {
    operation::Metadata metadata_;
};

template <Operation operation>
class OperatorNode : MetadataBase<operation>, public INode {
  private:
    std::array<INode *, operation::ARITY> inputs_;

    /// Parameters needed for forward pass
    operation::ForwardParams fp_;

    /// Parameters needed for backward pass
    operation::BackwardParams bp_;

    operation::Dispatch functions_;

    bool is_evaluated_ = false;

  public:
    OperatorNode(const std::array<INode *, operation::ARITY> &inputs)
        requires(!HasMetadata<operation>::value)
        : INode(operation::make_res_shape(inputs)), inputs_(inputs),
          functions_(operation::dispatch(inputs, this)) {}

    template <Operation op = operation>
    OperatorNode(const std::array<INode *, operation::ARITY> &inputs,
                 op::Metadata &&mdata)
        requires(HasMetadata<operation>::value)
        : MetadataBase<operation>(std::move(mdata)),
          INode(operation::make_res_shape(inputs, this->metadata_)),
          inputs_(inputs),
          functions_(operation::dispatch(inputs, this, this->metadata_)) {}

    [[nodiscard]] const char *operation_name() const noexcept override {
        return operation::OPERATION_NAME;
    }

    [[nodiscard]] bool is_evaluated() const noexcept override {
        return this->is_evaluated_;
    }

    [[nodiscard]] bool is_input() const noexcept override { return false; }

    void init_params() override {

        if constexpr (HasMetadata<operation>::value) {

            this->fp_ = typename operation::ForwardParams(this->inputs_, this,
                                                          this->metadata_);
            this->bp_ = typename operation::BackwardParams(this->inputs_, this,
                                                           this->metadata_);

        } else {

            this->fp_ = typename operation::ForwardParams(this->inputs_, this);
            this->bp_ = typename operation::BackwardParams(this->inputs_, this);
        }
    }

    void reset() override {
        this->is_evaluated_ = false;

        std::fill_n(this->value.data, this->value.size, Scalar{});
        std::fill_n(this->gradient.data, this->gradient.size, Scalar{});
    }

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
};

} // namespace kaad
