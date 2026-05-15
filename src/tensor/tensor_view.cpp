#include "kaad/tensor/internal/print_tensor.hpp"
#include "kaad/tensor/internal/tensor_types.hpp"
#include <kaad/tensor/tensor_view.hpp>
#include <ostream>
#include <span>

namespace kaad {

TensorView::TensorView(ShapeView shape, StridesView strides,
                       std::span<value_type> elements)
    : shape_(shape), strides_(strides), elements_(elements) {}

[[nodiscard]] ShapeView TensorView::shape() const noexcept {
    return this->shape_;
}

[[nodiscard]] StridesView TensorView::strides() const noexcept {
    return this->strides_;
}

[[nodiscard]] std::span<const TensorView::value_type>
TensorView::elements() const noexcept {
    return this->elements_;
}

[[nodiscard]] TensorView::size_type TensorView::rank() const {
    return this->shape_.size();
}

[[nodiscard]] Extent TensorView::extent(size_type axis) const noexcept {
    return this->shape_[axis];
}

[[nodiscard]] bool TensorView::scalar() const noexcept {
    return this->rank() == 0;
}

[[nodiscard]] TensorView::iterator TensorView::begin() const noexcept {
    return {this->shape_, this->strides_, this->elements_, false};
}

[[nodiscard]] TensorView::iterator TensorView::end() noexcept {
    return {this->shape_, this->strides_, this->elements_, true};
}

[[nodiscard]] TensorView::size_type TensorView::size() const noexcept {
    return this->elements_.size();
}

[[nodiscard]] bool TensorView::empty() const noexcept {
    return this->elements_.empty();
}

std::ostream &operator<<(std::ostream &stream, const TensorView &tensor) {
    print_tensor_impl(stream, tensor.shape(), tensor.strides(),
                      tensor.elements());
    return stream;
}

} // namespace kaad
