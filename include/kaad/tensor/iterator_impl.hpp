#pragma once

#include <algorithm>                    // for copy
#include <iterator>                     // for bidirectional_iterator_tag
#include <kaad/enums.hpp>               // for MUTABILITY
#include <kaad/scalar.hpp>              // for Scalar
#include <kaad/tensor/tensor_types.hpp> // for ShapeView, StrideView
#include <span>                         // for span
#include <type_traits>                  // for conditional_t
#include <vector>                       // for vector

namespace kaad {

class Tensor;

template <MUTABILITY M> class IteratorImpl {
    static constexpr bool IS_MUT = M == MUTABLE;

    using value_type = std::conditional_t<IS_MUT, Scalar, const Scalar>;
    using pointer = value_type *;
    using reference = value_type &;

    using iterator_concept = std::bidirectional_iterator_tag;
    using iterator_category = std::bidirectional_iterator_tag;

  private:
    /// Per-dim coordinates of the current element.
    StaticVector<std::size_t> cords_;

    ShapeView shape_;
    StridesView strides_;

    std::span<value_type> elements_;

  public:
    IteratorImpl(StaticVector<std::size_t> cords, ShapeView shape,
                 StridesView strides, std::span<value_type> elements)
        : cords_(cords), shape_(shape), strides_(strides), elements_(elements) {
    }

    operator IteratorImpl<IMMUTABLE>() {
        return {this->cords_, this->shape_, this->strides_, this->elements_};
    }

    reference operator*() const {
        int idx = 0;
        for (std::size_t i = 0; i < this->shape_.size(); i++) {
            idx += this->cords_[i] * this->strides_[i];
        }
        return this->elements_.data()[idx];
    }

    IteratorImpl &operator++() {
        // if tensor is scalar
        if (this->shape_.empty()) {
            this->cords_[0] = 1;
            return *this;
        }

        int rank = static_cast<int>(this->shape_.size() - 1);

        this->cords_[rank]++;

        while (this->cords_[rank] >= this->shape_[rank]) {

            this->cords_[rank] = 0;
            rank--;
            if (rank >= 0) {
                this->cords_[rank]++;

            } else {
                // increment every cord but the last, so iterator points one
                // past end and return.
                std::ranges::copy(this->shape_, this->cords_.begin());
                for (std::size_t i = 0; i < this->shape_.size() - 1; i++) {
                    this->cords_[i]--;
                }
                break;
            }
        }

        return *this;
    }

    IteratorImpl operator++(int) {
        IteratorImpl old = *this;
        ++(*this);
        return old;
    }

    IteratorImpl &operator--() {
        int rank = static_cast<int>(this->shape_.size() - 1);

        this->cords_[rank]--;

        while (this->cords_[rank] + 1 == 0) {

            this->cords_[rank] = this->shape_[rank] - 1;
            if (rank >= 0) {
                rank--;
                this->cords_[rank]--;

            } else {
                // set cords to all 0 and return
                std::ranges::fill(this->cords_, 0);
                break;
            }
        }

        return *this;
    }

    IteratorImpl operator--(int) {
        IteratorImpl old = *this;
        --(*this);
        return old;
    }

    friend bool operator==(const IteratorImpl &lhs, const IteratorImpl &rhs) {

        return
            // check if shapes are the same
            (lhs.shape_.data() == rhs.shape_.data() &&
             lhs.shape_.size() == rhs.shape_.size()) &&
            // check if strides are the same
            (lhs.strides_.data() == rhs.strides_.data() &&
             lhs.strides_.size() == rhs.strides_.size()) &&
            // check if elements are the same
            (lhs.elements_.data() == rhs.elements_.data() &&
             lhs.elements_.size() == rhs.elements_.size()) &&
            // check if cords are equal
            std::ranges::equal(lhs.cords_, rhs.cords_);
    }

    friend bool operator!=(const IteratorImpl &lhs, const IteratorImpl &rhs) {
        return !(lhs == rhs);
    }
};

} // namespace kaad
