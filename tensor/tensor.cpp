#include "tensor.hpp"

#include "../exceptions.hpp" // for argument_error
#include "common.hpp"        // for kaad::detail::print_tensor
#include "tensor_view.hpp"   // for kaad::Tensor_view
#include <algorithm>         // for std::copy, std::max, std::fill
#include <initializer_list>  // for std::initializer_list
#include <iostream>          // for std::ostream
#include <numeric>           // for std::iota
#include <span>              // for std::span
#include <vector>            // for std::vector

namespace kaad {

static void compute_stride(std::vector<int> &stride, int &len,
                           const std::vector<int> &shape) {
    stride.resize(shape.size());

    len = 1;
    int i = shape.size() - 1;

    len *= shape[i];
    stride[i--] = 1;

    for (; i >= 0; i--) {
        len *= shape[i];

        stride[i] = shape[i + 1] * stride[i + 1];
    }

    for (std::size_t i = 0; i < stride.size(); i++) {
        if (shape[i] <= 1) {
            stride[i] = 0;
        }
    }
}

Tensor::Tensor() : shape_({0}), stride_({0}), elements_({}) {}

Tensor::Tensor(std::span<const int> shape)
    : shape_(shape.begin(), shape.end()), stride_(shape.size()) {

    int len = 1;
    compute_stride(this->stride_, len, this->shape_);

    this->elements_.resize(len);
}

Tensor::Tensor(std::span<const int> shape, std::span<const int> stride)
    : shape_(shape.begin(), shape.end()),
      stride_(stride.begin(), stride.end()) {
    int len = 1;
    for (int d : this->shape_) {
        len *= d;
    }

    this->elements_.resize(len);
}

Tensor::Tensor(std::span<const int> shape, std::span<Scalar> elements)
    : shape_(shape.begin(), shape.end()),
      elements_(elements.begin(), elements.end()) {
    int len;
    compute_stride(this->stride_, len, this->shape_);

    if (len != elements.size()) {
        throw argument_error(
            "length suggested by shape and length of elements dont match");
    }
}

Tensor Tensor::empty(std::initializer_list<int> shape) {
    return Tensor(std::span<const int>(shape.begin(), shape.end()));
}

Tensor Tensor::full(std::initializer_list<int> shape, Scalar fill_value) {
    Tensor out(std::span<const int>(shape.begin(), shape.end()));
    std::fill(out.begin(), out.end(), fill_value);

    return out;
}

Tensor Tensor::zeros(std::initializer_list<int> shape) {
    Tensor out(std::span<const int>(shape.begin(), shape.end()));
    std::fill(out.begin(), out.end(), 0);

    return out;
}

Tensor Tensor::ones(std::initializer_list<int> shape) {
    Tensor out(std::span<const int>(shape.begin(), shape.end()));
    std::fill(out.begin(), out.end(), 1);

    return out;
}

Tensor Tensor::sequential(std::initializer_list<int> shape,
                          Scalar starting_value) {
    Tensor out(std::span<const int>(shape.begin(), shape.end()));
    std::iota(out.begin(), out.end(), starting_value);

    return out;
}

Tensor Tensor::linspace(std::initializer_list<int> shape, Scalar start,
                        Scalar step) {
    Tensor out(std::span<const int>(shape.begin(), shape.end()));

    Scalar value = start;
    for (int i = 0; i < out.size(); i++) {

        out.data()[i] = value;
        value += step;
    }

    return out;
}

Tensor Tensor::rand(std::initializer_list<int> shape, Scalar min, Scalar max) {
    if (!Tensor::seeded_) {
        Tensor::gen_.seed(std::random_device{}());
        seeded_ = true;
    }
    std::uniform_real_distribution<Scalar> dist{min, max};

    Tensor out(std::span<const int>(shape.begin(), shape.end()));
    for (int i = 0; i < out.size(); i++) {
        out.data()[i] = dist(Tensor::gen_);
    }

    return out;
}

Tensor Tensor::randn(std::initializer_list<int> shape, Scalar mean,
                     Scalar std) {

    if (!Tensor::seeded_) {
        Tensor::gen_.seed(std::random_device{}());
        seeded_ = true;
    }
    std::normal_distribution<Scalar> dist{mean, std};

    Tensor out(std::span<const int>(shape.begin(), shape.end()));
    for (int i = 0; i < out.size(); i++) {
        out.data()[i] = dist(Tensor::gen_);
    }

    return out;
}

void Tensor::manual_seed(uint64_t seed) {
    Tensor::gen_.seed(seed);
    seeded_ = true;
}

Tensor::size_type Tensor::rank() const noexcept {
    return static_cast<size_type>(this->shape_.size());
}

const std::vector<int> &Tensor::shape() const noexcept { return this->shape_; }

const std::vector<int> &Tensor::stride() const noexcept {
    return this->stride_;
}

Tensor::iterator Tensor::begin() noexcept {
    return static_cast<iterator>(this->elements_.data());
}

Tensor::const_iterator Tensor::begin() const noexcept {
    return static_cast<const_iterator>(this->elements_.data());
}

Tensor::iterator Tensor::end() noexcept {
    return static_cast<iterator>(this->elements_.data() +
                                 this->elements_.size());
}

Tensor::const_iterator Tensor::end() const noexcept {
    return static_cast<const_iterator>(this->elements_.data() +
                                       this->elements_.size());
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
