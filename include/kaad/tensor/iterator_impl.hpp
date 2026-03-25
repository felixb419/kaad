#pragma once

#include <algorithm>                    // for copy
#include <iterator>                     // for bidirectional_iterator_tag
#include <kaad/scalar.hpp>              // for Scalar
#include <kaad/tensor/tensor_types.hpp> // for ShapeView, StrideView
#include <span>                         // for span
#include <type_traits>                  // for conditional_t
#include <vector>                       // for vector

namespace kaad {

class Tensor;

template <bool isConst> class IteratorImpl {
    using value_type = std::conditional_t<isConst, const Scalar, Scalar>;
    using pointer = value_type *;
    using reference = value_type &;

    using iterator_concept = std::bidirectional_iterator_tag;
    using iterator_category = std::bidirectional_iterator_tag;

  private:
    const Tensor *origin_;

    std::vector<int> cords_; ///< Per-dim coordinates of the current element.

    ShapeView shape_;
    StrideView stride_;

    std::span<value_type> elements_;

  public:
    IteratorImpl(const Tensor *origin, std::vector<int> &&cords,
                 ShapeView shape, StrideView stride,
                 std::span<value_type> elements)
        : origin_(origin), cords_(cords), shape_(shape), stride_(stride),
          elements_(elements) {}

    const Tensor *origin() { return this->origin_; }

    reference operator*() const {
        int idx = 0;
        for (std::size_t i = 0; i < this->shape_.size(); i++) {
            idx += this->cords_[i] * this->stride_[i];
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

        while (this->cords_[rank] < 0) {

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

    bool operator==(const IteratorImpl &other) const {
        return (&this->origin_ == &other.origin_) &&
               std::equal(this->cords_.begin(), this->cords_.end(),
                          other.cords_.begin());
    }

    bool operator!=(const IteratorImpl &other) const {
        return !(*this == other);
    }
};

} // namespace kaad
