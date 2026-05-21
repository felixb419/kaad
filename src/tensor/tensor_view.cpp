#include "kaad/tensor/internal/print_tensor.hpp"
#include "kaad/tensor/internal/tensor.hpp"
#include "kaad/tensor/internal/tensor_types.hpp"

#include <kaad/tensor/tensor_view.hpp>
#include <ostream>
#include <span>

namespace kaad {

TensorView::TensorView(const Tensor *tensor) : origin_(tensor) {}

[[nodiscard]] ShapeView TensorView::shape() const noexcept {
    return this->origin_->shape;
}

[[nodiscard]] StridesView TensorView::strides() const noexcept {
    return this->origin_->strides;
}

[[nodiscard]] std::span<const TensorView::value_type>
TensorView::elements() const noexcept {
    return {this->origin_->data, this->origin_->size};
}

[[nodiscard]] TensorView::size_type TensorView::rank() const {
    return this->origin_->shape.size();
}

[[nodiscard]] Extent TensorView::extent(size_type axis) const noexcept {
    return this->origin_->shape[axis];
}

[[nodiscard]] bool TensorView::scalar() const noexcept {
    return this->origin_->rank() == 0;
}

[[nodiscard]] bool TensorView::is_contiguous() const noexcept {
    return this->origin_->is_contiguous();
}

[[nodiscard]] TensorView::iterator TensorView::begin() const noexcept {
    return {this->origin_->shape, this->origin_->strides,
            std::span{this->origin_->data, this->origin_->size}, false};
}

[[nodiscard]] TensorView::iterator TensorView::end() noexcept {
    return {this->origin_->shape, this->origin_->strides,
            std::span{this->origin_->data, this->origin_->size}, true};
}

[[nodiscard]] TensorView::size_type TensorView::size() const noexcept {
    return this->origin_->size;
}

[[nodiscard]] bool TensorView::empty() const noexcept {
    return this->origin_->size == 0;
}

std::ostream &operator<<(std::ostream &stream, const TensorView &tensor) {
    print_tensor_impl(stream, tensor.shape(), tensor.strides(),
                      tensor.elements());
    return stream;
}

} // namespace kaad
