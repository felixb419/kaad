#include <kaad/tensor/tensor.hpp>

#include "../exceptions.hpp"             // for ArgumentError
#include "print_tensor.hpp"              // for print_tensor_impl
#include <algorithm>                     // for fill, __fill_fn, max
#include <iostream>                      // for char_traits, ostream
#include <kaad/scalar.hpp>               // for Scalar
#include <kaad/tensor/iterator_impl.hpp> // for IteratorImpl
#include <kaad/tensor/tensor_types.hpp>  // for Stride, Shape_view
#include <kaad/tensor/tensor_view.hpp>   // for TensorViewConst, TensorViewMut
#include <span>                          // for span
#include <string>                        // for operator+, basic_string
#include <utility>                       // for move
#include <vector>                        // for vector, allocator

namespace kaad {

template class IteratorImpl<false>;
template class IteratorImpl<true>;

Stride Tensor::compute_stride(Shape_view shape) {

    if (shape.empty()) {
        return Stride{};
    }

    Stride stride(shape.size());
    int idx = static_cast<int>(shape.size()) - 1;

    stride[idx--] = 1;

    for (; idx >= 0; idx--) {
        stride[idx] = shape[idx + 1] * stride[idx + 1];
    }

    for (size_type i = 0; i < stride.size(); i++) {
        if (shape[i] <= 1) {
            stride[i] = 0;
        }
    }

    return stride;
}

Tensor::size_type Tensor::compute_size(std::span<const int> shape) {

    if (shape.empty()) {
        return 1;
    }

    size_type len = 1;
    for (int dim : shape) {
        len *= dim;
    }
    return len;
}

Stride checked_stride(std::span<const int> stride, std::span<const int> shape) {

    if (shape.size() != stride.size()) {
        throw ArgumentError(
            "size of shape param (" + std::to_string(shape.size()) +
            ") and size of stride param (" + std::to_string(stride.size()) +
            ") need to be equal");
    }

    return {stride.begin(), stride.end()};
}

std::vector<Scalar> checked_elements(std::span<const Scalar> elements,
                                     std::span<const int> shape) {

    Tensor::size_type implied_len = Tensor::compute_size(shape);
    if (implied_len != elements.size()) {
        throw ArgumentError(
            "size of elements param: " + std::to_string(elements.size()) +
            ", size implied by shape param: " + std::to_string(implied_len));
    }

    return {elements.begin(), elements.end()};
}

Tensor::Tensor() : elements_{0} {}

Tensor::Tensor(std::span<const int> shape)
    : shape_(shape.begin(), shape.end()),
      stride_(Tensor::compute_stride(shape_)),
      elements_(Tensor::compute_size(shape_)) {}

Tensor::Tensor(std::span<const Scalar> elements)
    : shape_{static_cast<int>(elements.size())},
      stride_(Tensor::compute_stride(shape_)),
      elements_(elements.begin(), elements.end()) {}

Tensor::Tensor(std::span<const int> shape, std::span<const int> stride)
    : shape_(shape.begin(), shape.end()),
      stride_(checked_stride(stride, shape)),
      elements_(Tensor::compute_size(shape_)) {}

Tensor::Tensor(std::span<const int> shape, std::span<const Scalar> elements)
    : shape_(shape.begin(), shape.end()),
      stride_(Tensor::compute_stride(shape_)),
      elements_(checked_elements(elements, shape)) {}

Tensor::Tensor(std::span<const int> shape, std::span<const int> stride,
               std::span<const Scalar> elements)
    : shape_(shape.begin(), shape.end()),
      stride_(checked_stride(stride, shape)),
      elements_(checked_elements(elements, shape)) {}

Tensor Tensor::full(std::span<const int> shape, Scalar fill_value) {
    Tensor out(std::span<const int>(shape.begin(), shape.end()));
    std::fill(out.data(), out.data() + out.size(), fill_value);

    return out;
}

Tensor Tensor::zeros(std::span<const int> shape) {
    return Tensor(std::span<const int>(shape.begin(), shape.end()));
}

Tensor Tensor::ones(std::span<const int> shape) {
    Tensor out(std::span<const int>(shape.begin(), shape.end()));
    std::fill(out.data(), out.data() + out.size(), 1);

    return out;
}

Tensor Tensor::sequential(std::span<const int> shape, Scalar starting_value) {
    Tensor out(std::span<const int>(shape.begin(), shape.end()));

    for (auto it = out.begin(); it != out.end(); it++) {
        *it = starting_value++;
    }

    return out;
}

Tensor Tensor::linspace(std::span<const int> shape, Scalar start, Scalar step) {
    Tensor out(std::span<const int>(shape.begin(), shape.end()));

    Scalar value = start;
    for (auto it = out.begin(); it != out.end(); it++) {
        *it = value;
        value += step;
    }

    return out;
}

Tensor Tensor::rand(std::span<const int> shape, Scalar min, Scalar max) {

    std::uniform_real_distribution<Scalar> dist{min, max};

    Tensor out(std::span<const int>(shape.begin(), shape.end()));
    for (size_type i = 0; i < out.size(); i++) {
        out.data()[i] = dist(Tensor::get_rng());
    }

    return out;
}

Tensor Tensor::randn(std::span<const int> shape, Scalar mean, Scalar std) {

    std::normal_distribution<Scalar> dist{mean, std};

    Tensor out(std::span<const int>(shape.begin(), shape.end()));
    for (size_type i = 0; i < out.size(); i++) {
        out.data()[i] = dist(Tensor::get_rng());
    }

    return out;
}

void Tensor::rng_seed(uint64_t seed) { Tensor::get_rng().seed(seed); }

Tensor::size_type Tensor::rank() const noexcept {
    return static_cast<size_type>(this->shape_.size());
}

std::span<const int> Tensor::shape() const noexcept { return this->shape_; }

std::span<const int> Tensor::stride() const noexcept { return this->stride_; }

std::span<Scalar> Tensor::elements() noexcept { return this->elements_; }

std::span<const Scalar> Tensor::elements() const noexcept {
    return this->elements_;
}

Tensor::iterator Tensor::begin() {
    std::vector<int> cords(std::max(this->rank(), static_cast<size_type>(1)));
    std::ranges::fill(cords, 0);

    return {this, std::move(cords), this->shape(), this->stride(),
            this->elements()};
}

Tensor::const_iterator Tensor::begin() const {
    std::vector<int> cords(std::max(this->rank(), static_cast<size_t>(1)));
    std::ranges::fill(cords, 0);

    return {this, std::move(cords), this->shape(), this->stride(),
            this->elements()};
}

Tensor::iterator Tensor::end() {

    std::vector<int> cords(this->shape_);

    if (this->rank() == 0) {

        cords.resize(1);
        cords[0] = 0;
    } else {

        // increment every cord but the last, so iterator points one past end.
        for (size_type i = 0; i < this->rank() - 1; i++) {
            cords[i]--;
        }
    }

    return {this, std::move(cords), this->shape(), this->stride(),
            this->elements()};
}

Tensor::const_iterator Tensor::end() const {
    std::vector<int> cords(this->shape_);

    if (this->rank() == 0) {

        cords.resize(1);
        cords[0] = 0;
    } else {

        // increment every cord but the last, so iterator points one past end.
        for (size_type i = 0; i < this->rank() - 1; i++) {
            cords[i]--;
        }
    }

    return {this, std::move(cords), this->shape(), this->stride(),
            this->elements()};
}

Tensor::size_type Tensor::size() const noexcept {
    return static_cast<size_type>(this->elements_.size());
}

bool Tensor::empty() const noexcept { return this->elements_.empty(); }

Tensor::reference Tensor::front() noexcept { return *this->begin(); }

Tensor::const_reference Tensor::front() const noexcept {
    return *this->begin();
}

Tensor::reference Tensor::back() noexcept { return *(this->end()--); }

Tensor::const_reference Tensor::back() const noexcept {
    return *(this->end()--);
}

Tensor::pointer Tensor::data() noexcept {
    return static_cast<pointer>(this->elements_.data());
}

Tensor::const_pointer Tensor::data() const noexcept {
    return static_cast<const_pointer>(this->elements_.data());
}

TensorViewConst Tensor::view() const noexcept {
    return {this->shape(), this->stride(), this->elements()};
}

TensorViewMut Tensor::view_mut() noexcept {
    return {this->shape(), this->stride(), this->elements()};
}

std::ostream &operator<<(std::ostream &stream, const Tensor &tensor) {

    print_tensor_impl(stream, tensor.shape(), tensor.stride(),
                      tensor.elements());

    return stream;
}

} // namespace kaad
