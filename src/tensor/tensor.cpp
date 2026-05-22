#include "kaad/tensor/internal/tensor.hpp"
#include "kaad/tensor/internal/print_tensor.hpp"
#include "kaad/tensor/internal/tensor_types.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <kaad/exceptions.hpp>
#include <kaad/static_vector.hpp>
#include <kaad/tensor/tensor_view.hpp>
#include <span>
#include <string>

namespace kaad {

Strides Tensor::compute_strides(ShapeView shape) noexcept {

    if (shape.empty()) {
        return Strides{};
    }

    Strides strides(shape.size(), Strides::UNCHECKED);
    int idx = static_cast<int>(shape.size()) - 1;

    strides[idx--] = 1;

    for (; idx >= 0; idx--) {
        strides[idx] = shape[idx + 1] * strides[idx + 1];
    }

    for (std::size_t i = 0; i < strides.size(); i++) {
        if (shape[i] <= 1) {
            strides[i] = 0;
        }
    }

    return strides;
}

std::size_t Tensor::compute_size(ShapeView shape) {

    if (shape.empty()) {
        return 1;
    }

    std::size_t len = 1;
    for (Extent ext : shape) {
        len *= ext;
    }
    return len;
}

Strides checked_strides(StridesView strides, ShapeView shape) {

    if (shape.size() != strides.size()) {
        throw ShapeError("size of shape param (" +
                         std::to_string(shape.size()) +
                         ") and size of strides param (" +
                         std::to_string(strides.size()) + ") need to be equal");
    }

    return {strides};
}

Tensor::Tensor(ShapeView shape)
    : shape(shape), strides(Tensor::compute_strides(shape)),
      size(Tensor::compute_size(shape)) {}

Tensor::Tensor(ShapeView shape, StridesView strides)
    : shape(shape), strides(checked_strides(strides, shape)),
      size(Tensor::compute_size(shape)) {}

TensorView Tensor::view() const noexcept { return {this}; }

std::size_t Tensor::rank() const noexcept {
    return static_cast<std::size_t>(this->shape.size());
}

bool Tensor::scalar() const noexcept { return this->rank() == 0; }

Tensor Tensor::transpose(StaticVector<std::size_t> perm) const {

    // if perm is not given, its filled so that shape is reversed
    if (perm.empty()) {

        perm.resize(this->rank());

        std::size_t val = this->rank();
        for (auto &elem : perm) {
            elem = --val;
        }
    }

    assert(perm.size() == this->rank());

    Tensor out;
    out.data = this->data;
    out.size = this->size;

    for (std::size_t i = 0; i < this->rank(); i++) {
        out.shape[i] = this->shape[perm[i]];
        out.strides[i] = this->strides[perm[i]];
    }

    return out;
}

Tensor Tensor::transpose_2d() const {

    Tensor out(*this);

    std::swap(out.shape.from_back(0), out.shape.from_back(1));
    std::swap(out.strides.from_back(0), out.strides.from_back(1));

    return out;
}

std::ostream &operator<<(std::ostream &stream, const Tensor &tensor) {

    print_tensor_impl(stream, tensor.shape, tensor.strides,
                      std::span{tensor.data, tensor.size});

    return stream;
}

} // namespace kaad
