#include "../../include/kaad/tensor/tensor_view.hpp"
#include "../../include/kaad/tensor/common.hpp" // for print_tensor
#include <vector>                               // for vector

namespace kaad {

Tensor_view::Tensor_view() noexcept {}

Tensor_view::Tensor_view(const int *shape, const int *stride, size_type rank,
                         const value_type *elements, size_type len) noexcept
    : shape(shape), stride(stride), rank(rank), elements(elements), len(len) {}

std::ostream &operator<<(std::ostream &os, const Tensor_view &view) {
    if (view.rank == 0) {
        std::cout << "[]";
    } else {
        std::vector<int> cords(view.rank);
        int indent = 0;

        detail::print_tensor(os, cords, view.shape, view.stride, view.rank,
                             view.elements, view.len, 0, indent);
    }
    return os;
}

Tensor_view_mut::Tensor_view_mut() noexcept {}

Tensor_view_mut::Tensor_view_mut(const int *shape, const int *stride,
                                 size_type rank, value_type *elements,
                                 size_type len) noexcept
    : shape(shape), stride(stride), rank(rank), elements(elements), len(len) {}

Tensor_view Tensor_view_mut::make_immutable() noexcept {
    return Tensor_view(this->shape, this->stride, this->rank, this->elements,
                       this->len);
}

std::ostream &operator<<(std::ostream &os, const Tensor_view_mut &view) {
    if (view.rank == 0) {
        std::cout << "[]";
    } else {
        std::vector<int> cords(view.rank);
        int indent = 0;

        detail::print_tensor(os, cords, view.shape, view.stride, view.rank,
                             view.elements, view.len, 0, indent);
    }
    return os;
}

} // namespace kaad
