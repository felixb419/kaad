#include "tensor_view.hpp"
#include "common.hpp" // for kaad::detail::print_tensor
#include <iostream>   // for std::ostream, std::cout
#include <vector>     // for std::vector

namespace kaad {

Tensor_view::Tensor_view() {}

Tensor_view::Tensor_view(const int *shape, const int *stride, size_type nDims,
                         const value_type *elements, size_type len)
    : shape(shape), stride(stride), nDims(nDims), elements(elements), len(len) {
}

static inline std::ostream &operator<<(std::ostream &os,
                                       const Tensor_view &view) {
    if (view.nDims == 0) {
        std::cout << "[]";
    } else {
        std::vector<int> cords(view.nDims);
        int indent = 0;

        detail::print_tensor(os, cords, view.shape, view.stride, view.nDims,
                             view.elements, view.len, 0, indent);
    }
    return os;
}

Tensor_view_mut::Tensor_view_mut() {}

Tensor_view_mut::Tensor_view_mut(const int *shape, const int *stride,
                                 size_type nDims, value_type *elements,
                                 size_type len)
    : shape(shape), stride(stride), nDims(nDims), elements(elements), len(len) {
}

Tensor_view Tensor_view_mut::make_immutable() {
    return Tensor_view(this->shape, this->stride, this->nDims, this->elements,
                       this->len);
}

static inline std::ostream &operator<<(std::ostream &os,
                                       const Tensor_view_mut &view) {
    if (view.nDims == 0) {
        std::cout << "[]";
    } else {
        std::vector<int> cords(view.nDims);
        int indent = 0;

        detail::print_tensor(os, cords, view.shape, view.stride, view.nDims,
                             view.elements, view.len, 0, indent);
    }
    return os;
}

} // namespace kaad
