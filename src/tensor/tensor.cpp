#include "../../include/kaad/tensor/tensor.hpp"
#include "../../include/kaad/exceptions.hpp"           // for argument_error
#include "../../include/kaad/scalar.hpp"               // for Scalar
#include "../../include/kaad/tensor/common.hpp"        // for print_tensor
#include "../../include/kaad/tensor/iterator_impl.hpp" // for iterator_impl
#include "../../include/kaad/tensor/tensor_view.hpp"   // for Tensor_view
#include <algorithm>                                   // for copy, fill, fill_n
#include <iostream>                                    // for char_traits
#include <span>                                        // for span
#include <string>                                      // for operator+, to_s...
#include <utility>                                     // for move
#include <vector>                                      // for allocator, vector

namespace kaad {

template class iterator_impl<false>;
template class iterator_impl<true>;

std::vector<int> Tensor::compute_stride(std::span<const int> shape) {

    std::vector<int> stride(shape.size());
    int i = shape.size() - 1;

    stride[i--] = 1;

    for (; i >= 0; i--) {
        stride[i] = shape[i + 1] * stride[i + 1];
    }

    for (std::size_t i = 0; i < stride.size(); i++) {
        if (shape[i] <= 1) {
            stride[i] = 0;
        }
    }

    return stride;
}

std::size_t Tensor::compute_size(std::span<const int> shape) {
    std::size_t len = 1;
    for (const int d : shape) {
        len *= d;
    }
    return len;
}

std::vector<int> checked_stride(std::span<const int> stride,
                                std::span<const int> shape) {

    if (shape.size() != stride.size()) {
        throw argument_error(
            "size of shape param (" + std::to_string(shape.size()) +
            ") and size of stride param (" + std::to_string(stride.size()) +
            ") need to be equal");
    }

    return std::vector(stride.begin(), stride.end());
}

std::vector<Scalar> checked_elements(std::span<const Scalar> elements,
                                     std::span<const int> shape) {

    std::size_t implied_len = Tensor::compute_size(shape);
    if (implied_len != elements.size()) {
        throw argument_error(
            "size of elements param: " + std::to_string(elements.size()) +
            ", size implied by shape param: " + std::to_string(implied_len));
    }

    return std::vector<Scalar>(elements.begin(), elements.end());
}

Tensor::Tensor() : shape_({0}), stride_({0}), elements_({}) {}

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

Tensor Tensor::empty(std::span<const int> shape) {
    return Tensor(std::span<const int>(shape.begin(), shape.end()));
}

Tensor Tensor::full(std::span<const int> shape, Scalar fill_value) {
    Tensor out(std::span<const int>(shape.begin(), shape.end()));
    std::fill(out.data(), out.data() + out.size(), fill_value);

    return out;
}

Tensor Tensor::zeros(std::span<const int> shape) {
    Tensor out(std::span<const int>(shape.begin(), shape.end()));
    std::fill(out.data(), out.data() + out.size(), 0);

    return out;
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
    if (!Tensor::seeded_) {
        Tensor::gen_.seed(std::random_device{}());
        seeded_ = true;
    }
    std::uniform_real_distribution<Scalar> dist{min, max};

    Tensor out(std::span<const int>(shape.begin(), shape.end()));
    for (std::size_t i = 0; i < out.size(); i++) {
        out.data()[i] = dist(Tensor::gen_);
    }

    return out;
}

Tensor Tensor::randn(std::span<const int> shape, Scalar mean, Scalar std) {

    if (!Tensor::seeded_) {
        Tensor::gen_.seed(std::random_device{}());
        seeded_ = true;
    }
    std::normal_distribution<Scalar> dist{mean, std};

    Tensor out(std::span<const int>(shape.begin(), shape.end()));
    for (std::size_t i = 0; i < out.size(); i++) {
        out.data()[i] = dist(Tensor::gen_);
    }

    return out;
}

void Tensor::manual_seed(uint64_t seed) {
    Tensor::gen_.seed(seed);
    seeded_ = true;
}

void Tensor::reshape(std::span<const int> shape) {

    std::copy(shape.begin(), shape.end(), this->shape_.begin());

    std::size_t suggested_len = compute_size(this->shape_);
    this->stride_ = compute_stride(this->shape_);

    if (suggested_len != this->size()) {
        throw argument_error("length suggested by shape and previous length "
                             "number of elements dont match");
    }
}

Tensor::size_type Tensor::rank() const noexcept {
    return static_cast<size_type>(this->shape_.size());
}

const std::vector<int> &Tensor::shape() const noexcept { return this->shape_; }

const std::vector<int> &Tensor::stride() const noexcept {
    return this->stride_;
}

Tensor::iterator Tensor::begin() {
    std::vector<int> cords(this->rank());
    std::fill(cords.begin(), cords.end(), 0);

    return iterator(*this, std::move(cords));
}

Tensor::const_iterator Tensor::begin() const {
    std::vector<int> cords(this->rank());
    std::fill(cords.begin(), cords.end(), 0);

    return const_iterator(*this, std::move(cords));
}

Tensor::iterator Tensor::end() {

    std::vector<int> cords(this->shape_);
    // increment every cord but the last, so iterator points one past end.
    for (size_type i = 0; i < this->rank() - 1; i++) {
        cords[i]--;
    }

    return iterator(*this, std::move(cords));
}

Tensor::const_iterator Tensor::end() const {
    std::vector<int> cords(this->shape_);
    // increment every cord but the last, so iterator points one past end.
    for (size_type i = 0; i < this->rank() - 1; i++) {
        cords[i]--;
    }

    return const_iterator(*this, std::move(cords));
}

Tensor::size_type Tensor::size() const noexcept {
    return static_cast<size_type>(this->elements_.size());
}

bool Tensor::empty() const noexcept { return this->elements_.empty(); }

Tensor::reference Tensor::front() noexcept {
    return static_cast<reference>(this->elements_.front());
}

Tensor::const_reference Tensor::front() const noexcept {
    return static_cast<const_reference>(this->elements_.front());
}

Tensor::reference Tensor::back() noexcept {
    return static_cast<reference>(this->elements_.back());
}

Tensor::const_reference Tensor::back() const noexcept {
    return static_cast<const_reference>(this->elements_.back());
}

Tensor::pointer Tensor::data() noexcept {
    return static_cast<pointer>(this->elements_.data());
}

Tensor::const_pointer Tensor::data() const noexcept {
    return static_cast<const_pointer>(this->elements_.data());
}

struct Tensor_view Tensor::view() const noexcept {
    return Tensor_view(this->shape_.data(), this->stride_.data(), this->rank(),
                       this->data(), this->size());
}

struct Tensor_view_mut Tensor::view_mut() noexcept {
    return Tensor_view_mut(this->shape_.data(), this->stride_.data(),
                           this->rank(), this->elements_.data(), this->size());
}

std::ostream &operator<<(std::ostream &os, const Tensor &tensor) {
    if (tensor.rank() == 0 || tensor.size() == 0) {
        os << "[]";
    } else {
        std::vector<int> cords(tensor.rank());
        int indent = 0;

        detail::print_tensor(os, cords, tensor.shape().data(),
                             tensor.stride().data(), tensor.rank(),
                             tensor.data(), tensor.size(), 0, indent);
    }
    return os;
}

} // namespace kaad
