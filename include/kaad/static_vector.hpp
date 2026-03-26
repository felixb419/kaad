#pragma once

#include <array>             // for array
#include <cassert>           // for assert
#include <kaad/max_rank.hpp> // for KAAD_MAX_RANK
#include <span>              // for span

namespace kaad {

/**
 * @brief Struct to represent stack-allocated vector,
 * size is limited to @c KAAD_MAX_RANK.
 * @tparam T Element type
 */
template <typename T> struct StaticVector {

    using value_type = T;
    using reference = value_type &;
    using const_reference = const value_type &;
    using pointer = value_type *;
    using const_pointer = const value_type *;
    using size_type = std::size_t;
    using iterator = value_type *;
    using const_iterator = const value_type *;

    using view_type = std::span<const value_type>;

    std::array<value_type, KAAD_MAX_RANK> elements_;

    std::size_t size_ = 0;

    StaticVector() = default;

    /// @note Elements are value initialized.
    StaticVector(size_type size) {

        assert(size <= KAAD_MAX_RANK);

        this->size_ = size;
        std::fill(this->elements_.data(), this->elements_.data() + size,
                  value_type{});
    }

    StaticVector(const_pointer begin, size_type size) {

        assert(size <= KAAD_MAX_RANK);

        this->size_ = size;
        std::copy(begin, begin + size, this->elements_.data());
    }

    StaticVector(const_pointer first, const_pointer last) {

        assert(first <= last);
        assert(size_ <= KAAD_MAX_RANK);

        this->size_ = last - first;
        std::copy(first, last, elements_.data());
    }

    template <std::ranges::input_range IR> StaticVector(IR init) {
        auto begin = std::ranges::begin(init);
        auto end = std::ranges::end(init);
        std::size_t size = end - begin;

        assert(begin <= end);
        assert(size <= KAAD_MAX_RANK);

        this->size_ = size;
        std::copy(begin, end, this->elements_.data());
    }

    [[nodiscard]] size_type size() const noexcept { return this->size_; }

    /// @return KAAD_MAX_RANK.
    [[nodiscard]] size_type max_size() const noexcept { return KAAD_MAX_RANK; }

    [[nodiscard]] bool empty() const noexcept { return this->size_ == 0; }

    [[nodiscard]] const_reference operator[](size_type idx) const {

        assert(idx < this->size_);

        return this->elements_[idx];
    }

    reference operator[](size_type idx) {

        assert(idx < this->size_);

        return this->elements_[idx];
    }

    /// @note If new elements are created they are value initialized.
    void resize(size_type count) {

        assert(count <= KAAD_MAX_RANK);

        if (count > this->size_) {
            std::fill(this->elements_.data() + this->size_,
                      this->elements_.data() + count, value_type{});
        }

        this->size_ = count;
    }

    void push_back(value_type value) {

        assert(this->size_ + 1 <= KAAD_MAX_RANK);

        this->elements_[this->size_] = value;
        this->size_++;
    }

    [[nodiscard]] pointer data() noexcept { return this->elements_.data(); }

    [[nodiscard]] const_pointer data() const noexcept {
        return this->elements_.data();
    }

    [[nodiscard]] iterator begin() noexcept { return this->elements_.data(); }

    [[nodiscard]] const_iterator begin() const noexcept {
        return this->elements_.data();
    }

    [[nodiscard]] iterator end() noexcept {
        return this->elements_.data() + this->size_;
    }

    [[nodiscard]] const_iterator end() const noexcept {
        return this->elements_.data() + this->size_;
    }

    [[nodiscard]] view_type view() const noexcept {
        return std::span<value_type>(elements_.data(), this->size_);
    }
};

} // namespace kaad
