#include <algorithm> // for fill
#include <cstddef>
#include <cstdint>
#include <iostream>                              // for char_traits, ostream
#include <kaad/exceptions.hpp>                   // for ShapeError
#include <kaad/scalar.hpp>                       // for Scalar
#include <kaad/tensor/internal/print_tensor.hpp> // for print_tensor_impl
#include <kaad/tensor/internal/tensor_types.hpp> // for ShapeView, Strides, StridesView
#include <kaad/tensor/tensor.hpp>
#include <kaad/tensor/tensor_view.hpp> // for TensorViewConst, TensorViewMut
#include <random>
#include <span>   // for span
#include <string> // for operator+, basic_string, to_...
#include <vector> // for allocator, vector

namespace kaad {

Strides Tensor::compute_strides(ShapeView shape) {

    if (shape.empty()) {
        return Strides{};
    }

    Strides strides(shape.size());
    int idx = static_cast<int>(shape.size()) - 1;

    strides[idx--] = 1;

    for (; idx >= 0; idx--) {
        strides[idx] = shape[idx + 1] * strides[idx + 1];
    }

    for (size_type i = 0; i < strides.size(); i++) {
        if (shape[i] <= 1) {
            strides[i] = 0;
        }
    }

    return strides;
}

Tensor::size_type Tensor::compute_size(ShapeView shape) {

    if (shape.empty()) {
        return 1;
    }

    size_type len = 1;
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
    : shape_(shape), strides_(Tensor::compute_strides(shape_)),
      elements_(Tensor::compute_size(shape_)) {}

Tensor::Tensor(std::span<const Scalar> elements)
    : shape_(static_cast<std::size_t>(elements.size())),
      strides_(Tensor::compute_strides(shape_)),
      elements_(elements.begin(), elements.end()) {}

Tensor::Tensor(ShapeView shape, StridesView strides)
    : shape_(shape), strides_(checked_strides(strides, shape)),
      elements_(Tensor::compute_size(shape_)) {}

Tensor::Tensor(ShapeView shape, std::span<const Scalar> elements)
    : shape_(shape), strides_(Tensor::compute_strides(shape_)),
      elements_(checked_elements(elements, shape)) {}

Tensor::Tensor(ShapeView shape, StridesView strides,
               std::span<const Scalar> elements)
    : shape_(shape), strides_(checked_strides(strides, shape)),
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

[[nodiscard]] Extent Tensor::extent(size_type axis) const noexcept {
    return this->shape_[axis];
}

StridesView Tensor::strides() const noexcept { return this->strides_; }

std::span<Scalar> Tensor::elements() noexcept { return this->elements_; }

std::span<const Scalar> Tensor::elements() const noexcept {
    return this->elements_;
}

bool Tensor::scalar() const noexcept { return this->rank() == 0; }

Tensor::iterator Tensor::begin() noexcept {
    return {this->shape_, this->strides_, this->elements_, false};
}

Tensor::const_iterator Tensor::begin() const noexcept {
    return {this->shape_, this->strides_, this->elements_, false};
}

Tensor::iterator Tensor::end() noexcept {
    return {this->shape_, this->strides_, this->elements_, true};
}

Tensor::const_iterator Tensor::end() const noexcept {
    return {this->shape_, this->strides_, this->elements_, true};
}

Tensor::size_type Tensor::size() const noexcept {
    return static_cast<size_type>(this->elements_.size());
}

bool Tensor::empty() const noexcept { return this->elements_.empty(); }

Tensor::pointer Tensor::data() noexcept {
    return static_cast<pointer>(this->elements_.data());
}

Tensor::const_pointer Tensor::data() const noexcept {
    return static_cast<const_pointer>(this->elements_.data());
}

TensorViewConst Tensor::view() const noexcept {
    return {this->shape_, this->strides_, this->elements()};
}

TensorViewMut Tensor::view_mut() noexcept {
    return {this->shape_, this->strides_, this->elements()};
}

std::ostream &operator<<(std::ostream &stream, const Tensor &tensor) {

    print_tensor_impl(stream, tensor.shape(), tensor.strides(),
                      tensor.elements());

    return stream;
}

} // namespace kaad
