#pragma once

#include <algorithm>              // for copy
#include <iterator>               // for bidirectional_iterator_tag
#include <kaad/scalar.hpp>        // for Scalar
#include <kaad/tensor/tensor.hpp> // for Shape_view, Stride_view
#include <span>                   // for span
#include <type_traits>            // for conditional_t
#include <vector>                 // for vector

namespace kaad {

template <bool isConst> class IteratorImpl {
  private:
    using Tensor_reference =
        std::conditional_t<isConst, const Tensor &, Tensor &>;
    Tensor_reference origin_;    ///< Origin tensor of the iterator.
    std::vector<int> cords_;     ///< Per-dim coordinates of the element.
    Tensor::Shape_view shape_;   ///< Shape of the tensor.
    Tensor::Stride_view stride_; ///< Stride array of the tensor.

  public:
    using iterator_concept = std::bidirectional_iterator_tag;
    using iterator_category = std::bidirectional_iterator_tag;
    using value_type = Scalar;
    using difference_type = std::ptrdiff_t;
    using pointer = std::conditional_t<isConst, const Scalar *, Scalar *>;
    using reference = std::conditional_t<isConst, const Scalar &, Scalar &>;

    IteratorImpl(Tensor_reference &origin, std::vector<int> &&cords)
        : origin_(origin), cords_(cords), shape_(origin_.shape()),
          stride_(origin_.stride()) {}

    const Tensor &origin() { return this->origin_; }

    reference operator*() const {
        int idx = 0;
        for (std::size_t i = 0; i < this->origin_.rank(); i++) {
            idx += this->cords_[i] * this->stride_[i];
        }
        return this->origin_.data()[idx];
    }

    IteratorImpl &operator++() {
        if (this->origin_.rank() == 0) {
            this->cords_[0] = 1;
            return *this;
        }

        int rank = static_cast<int>(this->origin_.rank() - 1);

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
                for (std::size_t i = 0; i < this->origin_.rank() - 1; i++) {
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
        int rank = static_cast<int>(this->origin_.rank() - 1);

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
