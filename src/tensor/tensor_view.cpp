#include "../../include/kaad/tensor/tensor_view.hpp"
#include "../../include/kaad/tensor/common.hpp" // for print_tensor_impl
#include "kaad/scalar.hpp"                      // for Scalar
#include <span>                                 // for span

namespace kaad {

Tensor_view::Tensor_view() noexcept = default;

Tensor_view::Tensor_view(const int *shape, const int *stride, size_type rank,
                         const value_type *elements, size_type len) noexcept
    : shape(shape), stride(stride), rank(rank), elements(elements), len(len) {}

std::ostream &operator<<(std::ostream &stream, const Tensor_view &view) {
    print_tensor_impl(stream, std::span<const int>(view.shape, view.rank),
                      std::span<const int>(view.stride, view.rank),
                      std::span<const Scalar>(view.elements, view.len));

    return stream;
}

Tensor_view_mut::Tensor_view_mut() noexcept = default;

Tensor_view_mut::Tensor_view_mut(const int *shape, const int *stride,
                                 size_type rank, value_type *elements,
                                 size_type len) noexcept
    : shape(shape), stride(stride), rank(rank), elements(elements), len(len) {}

Tensor_view Tensor_view_mut::make_immutable() const noexcept {
    return {this->shape, this->stride, this->rank, this->elements, this->len};
}

std::ostream &operator<<(std::ostream &stream, const Tensor_view_mut &view) {

    stream << view.make_immutable();

    return stream;
}

} // namespace kaad
