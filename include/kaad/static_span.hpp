#pragma once

#include <cassert>
#include <cstddef>
#include <kaad/exceptions.hpp>
#include <kaad/max_rank.hpp>
#include <kaad/scalar.hpp>

namespace kaad {

template <typename T> struct StaticVector;

template <typename T> struct StaticSpan {
  public:
    using value_type = const T; ///< Immutable by default.
    using reference = value_type &;
    using pointer = value_type *;
    using size_type = std::size_t;
    using iterator = value_type *;

    static constexpr const std::size_t MAX_SIZE = KAAD_MAX_RANK;

  private:
    pointer data_;   ///< First element
    size_type size_; ///< Number of elements

  public:
    StaticSpan() = default;

    StaticSpan(pointer begin, size_type size) {

        if (size > KAAD_MAX_RANK) {
            throw CapacityError("size param must not be larger than maximum "
                                "size (KAAD_MAX_RANK)");
        }

        this->data_ = begin;
        this->size_ = size;
    }

    StaticSpan(pointer first, pointer last) {

        if (first > last) {
            throw ArgumentError("first pointer must not be after last");
        }
        if ((last - first) > KAAD_MAX_RANK) {
            throw CapacityError("distance between first and last must not be"
                                "bigger than maximum size (KAAD_MAX_RANK)");
        }

        this->data_ = first;
        this->size_ = std::distance(first, last);
    }

    StaticSpan(const StaticVector<T> &init) noexcept
        : data_(init.data()), size_(init.size()) {}

    [[nodiscard]] size_type size() const noexcept { return this->size_; }

    /// @return KAAD_MAX_RANK.
    [[nodiscard]] size_type max_size() const noexcept { return KAAD_MAX_RANK; }

    /// @return True if vector is empty false otherwise.
    [[nodiscard]] bool empty() const noexcept { return this->size_ == 0; }

    [[nodiscard]] reference operator[](size_type idx) const {

        assert(idx < this->size_);

        return this->data_[idx];
    }

    reference operator[](size_type idx) {

        assert(idx < this->size_);

        return this->data_[idx];
    }

    /// @return The element at index @c size() - 1 - @p idx.
    reference from_back(size_type idx) {
        return this->data_[this->size_ - 1 - idx];
    }

    [[nodiscard]] pointer data() noexcept { return this->data_; }

    [[nodiscard]] pointer data() const noexcept { return this->data_; }

    [[nodiscard]] iterator begin() noexcept { return this->data_; }

    [[nodiscard]] iterator begin() const noexcept { return this->data_; }

    [[nodiscard]] iterator end() noexcept { return this->data_ + this->size_; }

    [[nodiscard]] iterator end() const noexcept {
        return this->data_ + this->size_;
    }

    friend class TensorView;
};

} // namespace kaad
