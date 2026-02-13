#include "tensor.hpp"
#include "common.hpp"       // for kaad::detail::print_tensor
#include "tensor_view.hpp"  // for kaad::Tensor_view
#include <algorithm>        // for std::copy, std::max, std::fill
#include <initializer_list> // for std::initializer_list
#include <iostream>         // for std::ostream
#include <vector>           // for std::vector

namespace kaad {

Tensor::Tensor() {}

Tensor::Tensor(std::initializer_list<int> shape, value_type fill)
    : shape_(shape.begin(), shape.end()) {
    int len;
    detail::compute_stride(this->stride_, len, this->shape_);

    this->elements_.resize(len);
    std::fill(this->elements_.begin(), this->elements_.end(), fill);
}

Tensor::Tensor(value_type scalar) : shape_(1), stride_(1), elements_(1) {
    this->shape_[0] = 1;
    this->stride_[0] = 0;
    this->elements_[0] = scalar;
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
