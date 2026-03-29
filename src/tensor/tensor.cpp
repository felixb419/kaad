#include <kaad/tensor/tensor.hpp>

#include "kaad/max_rank.hpp"             // for KAAD_MAX_RANK
#include "print_tensor.hpp"              // for print_tensor_impl
#include <algorithm>                     // for fill, max
#include <iostream>                      // for char_traits, ostream
#include <kaad/exceptions.hpp>           // for ShapeError
#include <kaad/scalar.hpp>               // for Scalar
#include <kaad/static_vector.hpp>        // for StaticVector
#include <kaad/tensor/iterator_impl.hpp> // for IteratorImpl
#include <kaad/tensor/tensor_types.hpp>  // for ShapeView, Stride, StrideView
#include <kaad/tensor/tensor_view.hpp>   // for TensorViewConst, TensorViewMut
#include <span>                          // for span
#include <string>                        // for operator+, basic_string
#include <vector>                        // for allocator, vector

namespace kaad {

Stride Tensor::compute_stride(ShapeView shape) {

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

Tensor::size_type Tensor::compute_size(ShapeView shape) {

    if (shape.empty()) {
        return 1;
    }

    size_type len = 1;
    for (int dim : shape) {
        len *= dim;
    }
    return len;
}

Stride checked_stride(StrideView stride, ShapeView shape) {

    if (shape.size() != stride.size()) {
        throw ShapeError("size of shape param (" +
                         std::to_string(shape.size()) +
                         ") and size of stride param (" +
                         std::to_string(stride.size()) + ") need to be equal");
    }

    return {stride};
}

std::vector<Scalar> checked_elements(std::span<const Scalar> elements,
                                     ShapeView shape) {

    Tensor::size_type implied_len = Tensor::compute_size(shape);
    if (implied_len != elements.size()) {
        throw ShapeError("size of elements param (" +
                         std::to_string(elements.size()) +
                         ") must be equal to size implied by shape param (" +
                         std::to_string(implied_len) + ")");
    }

    return {elements.begin(), elements.end()};
}

Tensor::Tensor() : elements_{0} {}

Tensor::Tensor(ShapeView shape)
    : shape_(shape), stride_(Tensor::compute_stride(shape_)),
      elements_(Tensor::compute_size(shape_)) {}

Tensor::Tensor(std::span<const Scalar> elements)
    : shape_(static_cast<std::size_t>(elements.size())),
      stride_(Tensor::compute_stride(shape_)),
      elements_(elements.begin(), elements.end()) {}

Tensor::Tensor(ShapeView shape, StrideView stride)
    : shape_(shape), stride_(checked_stride(stride, shape)),
      elements_(Tensor::compute_size(shape_)) {}

Tensor::Tensor(ShapeView shape, std::span<const Scalar> elements)
    : shape_(shape), stride_(Tensor::compute_stride(shape_)),
      elements_(checked_elements(elements, shape)) {}

Tensor::Tensor(ShapeView shape, StrideView stride,
               std::span<const Scalar> elements)
    : shape_(shape), stride_(checked_stride(stride, shape)),
      elements_(checked_elements(elements, shape)) {}

Tensor Tensor::full(ShapeView shape, Scalar fill_value) {
    Tensor out(ShapeView(shape.begin(), shape.end()));
    std::fill(out.data(), out.data() + out.size(), fill_value);

    return out;
}

Tensor Tensor::zeros(ShapeView shape) {
    return Tensor(ShapeView(shape.begin(), shape.end()));
}

Tensor Tensor::ones(ShapeView shape) {
    Tensor out(ShapeView(shape.begin(), shape.end()));
    std::fill(out.data(), out.data() + out.size(), 1);

    return out;
}

Tensor Tensor::sequential(ShapeView shape, Scalar starting_value) {
    Tensor out(ShapeView(shape.begin(), shape.end()));

    for (Scalar &elem : out) {
        elem = starting_value++;
    }

    return out;
}

Tensor Tensor::linspace(ShapeView shape, Scalar start, Scalar step) {
    Tensor out(ShapeView(shape.begin(), shape.end()));

    Scalar value = start;
    for (Scalar &elem : out) {
        elem = value;
        value += step;
    }

    return out;
}

Tensor Tensor::rand(ShapeView shape, Scalar min, Scalar max) {

    std::uniform_real_distribution<Scalar> dist{min, max};

    Tensor out(ShapeView(shape.begin(), shape.end()));
    for (size_type i = 0; i < out.size(); i++) {
        out.data()[i] = dist(Tensor::get_rng());
    }

    return out;
}

Tensor Tensor::randn(ShapeView shape, Scalar mean, Scalar std) {

    std::normal_distribution<Scalar> dist{mean, std};

    Tensor out(ShapeView(shape.begin(), shape.end()));
    for (size_type i = 0; i < out.size(); i++) {
        out.data()[i] = dist(Tensor::get_rng());
    }

    return out;
}

void Tensor::rng_seed(uint64_t seed) { Tensor::get_rng().seed(seed); }

Tensor::size_type Tensor::rank() const noexcept {
    return static_cast<size_type>(this->shape_.size());
}

ShapeView Tensor::shape() const noexcept { return this->shape_; }

StrideView Tensor::stride() const noexcept { return this->stride_; }

std::span<Scalar> Tensor::elements() noexcept { return this->elements_; }

std::span<const Scalar> Tensor::elements() const noexcept {
    return this->elements_;
}

Tensor::iterator Tensor::begin() noexcept {
    static_assert(KAAD_MAX_RANK >= 1);
    StaticVector<int> cords(std::max(this->rank(), static_cast<size_type>(1)),
                            StaticVector<int>::UNCHECKED);

    return {this, cords, this->shape_, this->stride_, this->elements()};
}

Tensor::const_iterator Tensor::begin() const noexcept {
    static_assert(KAAD_MAX_RANK >= 1);
    StaticVector<int> cords(std::max(this->rank(), static_cast<size_type>(1)),
                            StaticVector<int>::UNCHECKED);

    return {this, cords, this->shape_, this->stride_, this->elements()};
}

Tensor::iterator Tensor::end() noexcept {

    StaticVector<int> cords(this->shape_);

    if (this->rank() == 0) {

        static_assert(KAAD_MAX_RANK >= 1);
        cords.resize(1, StaticVector<int>::UNCHECKED);
        cords[0] = 0;
    } else {

        // increment every cord but the last, so iterator points one past end.
        for (size_type i = 0; i < this->rank() - 1; i++) {
            cords[i]--;
        }
    }

    return {this, cords, this->shape_, this->stride_, this->elements()};
}

Tensor::const_iterator Tensor::end() const noexcept {
    StaticVector<int> cords(this->shape_);

    if (this->rank() == 0) {

        static_assert(KAAD_MAX_RANK >= 1);
        cords.resize(1, StaticVector<int>::UNCHECKED);
        cords[0] = 0;
    } else {

        // increment every cord but the last, so iterator points one past end.
        for (size_type i = 0; i < this->rank() - 1; i++) {
            cords[i]--;
        }
    }

    return {this, cords, this->shape_, this->stride_, this->elements()};
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
    return {this->shape_, this->stride_, this->elements()};
}

TensorViewMut Tensor::view_mut() noexcept {
    return {this->shape_, this->stride_, this->elements()};
}

std::ostream &operator<<(std::ostream &stream, const Tensor &tensor) {

    print_tensor_impl(stream, tensor.shape(), tensor.stride(),
                      tensor.elements());

    return stream;
}

} // namespace kaad
