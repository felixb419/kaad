#pragma once

#include <algorithm>       // for copy
#include <iterator>        // for bidirectional_iterator_tag
#include <kaad/enums.hpp>  // for MUTABILITY
#include <kaad/scalar.hpp> // for Scalar
#include <kaad/tensor/internal/tensor_types.hpp> // for ShapeView, StrideView
#include <span>                                  // for span
#include <type_traits>                           // for conditional_t
#include <vector>                                // for vector

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
    /// Per-axis coordinates of the current element.
    StaticVector<std::size_t> coords_;

    pointer current_ = nullptr;

    bool is_end_ = false;

    ShapeView shape_;
    StridesView strides_;

    std::span<value_type> elements_;

    void set_to_begin() {

        std::ranges::fill(this->coords_, 0);

        this->current_ = &this->elements_.front();

        this->is_end_ = false;
    }

    void set_to_end() {

        std::size_t idx = 0;
        for (; idx < this->coords_.size(); idx++) {
            this->coords_[idx] = this->shape_[idx] - 1;
        }

        this->current_ = &this->elements_.back();

        this->is_end_ = true;
    }

  public:
    /// Constructs an iterator refering to the beginning or the end depending
    /// on @p is_end.
    IteratorImpl(ShapeView shape, StridesView strides,
                 std::span<value_type> elements, bool is_end)
        : coords_(shape.size(), StaticVector<std::size_t>::UNCHECKED),
          shape_(shape), strides_(strides), elements_(elements) {

        if (is_end) {
            this->set_to_end();
        } else {
            this->set_to_begin();
        }
    }

    operator IteratorImpl<IMMUTABLE>() {
        return {this->coords_, this->current_, this->is_end_,
                this->shape_,  this->strides_, this->elements_};
    }

    reference operator*() const { return *this->current_; }

    IteratorImpl &operator++() {

        if (this->is_end_) {
            return *this;
        }

        if (this->shape_.empty()) {
            this->is_end_ = true;
            return *this;
        }

        for (int axis = this->shape_.size() - 1; axis >= 0; axis--) {

            this->coords_[axis]++;
            this->current_ += this->strides_[axis];

            if (this->coords_[axis] < this->shape_[axis]) {

                return *(this);
            }

            // at end of axis
            this->coords_[axis] = 0;
            this->current_ -= this->shape_[axis] * this->strides_[axis];
        }

        this->set_to_end();
        return *this;
    }

    IteratorImpl operator++(int) {
        IteratorImpl old = *this;
        ++(*this);
        return old;
    }

    IteratorImpl &operator--() {

        if (this->is_end_) {
            this->is_end_ = false;
            return *this;
        }

        if (this->shape_.empty()) {
            return *this;
        }

        for (int axis = this->shape_.size() - 1; axis >= 0; axis--) {

            this->coords_[axis]--;
            this->current_ -= this->strides_[axis];

            if (this->coords_[axis] >= 0) {

                return *(this);
            }

            // at end of axis
            this->coords_[axis] = this->shape_[axis] - 1;
            this->current_ += this->shape_[axis] * this->strides_[axis];
        }

        this->set_to_begin();
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
            // check if coords are equal
            std::ranges::equal(lhs.coords_, rhs.coords_) &&
            // check end flags
            lhs.is_end_ == rhs.is_end_;
    }

    friend bool operator!=(const IteratorImpl &lhs, const IteratorImpl &rhs) {
        return !(lhs == rhs);
    }
};

} // namespace kaad
