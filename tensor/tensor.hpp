#pragma once

#include <algorithm>        // for std::copy, std::max, std::fill
#include <cstddef>          // for size_t
#include <initializer_list> // for std::initializer_list
#include <iostream>  // for std::operator<<, std::ostream, std::cout, std::end
#include <stdexcept> // for std::invalid_argument

#include "common.hpp" // for kaad::detail::print_tensor
#include "tView.hpp"  // for kaad::tView

namespace kaad {

/**
 * @brief A class representing a multi-dimensional tensor.
 *
 * This class supports creation, copying, moving, and indexing of tensors,
 * and provides utilities for memory management and shape/stride calculation.
 *
 * @tparam T The type of the values stored in the tensor.
 */
template <typename T> class Tensor {
  public:
    /// Pointer to the shape array, where shape[i] is the size along dimension
    /// i.
    int *shape = nullptr;
    /// Pointer to the stride array, where stride[i] is the step size to move
    /// along dimension i.
    int *stride = nullptr;
    /// Number of dimensions of the tensor.
    size_t nDims = 0;
    /// Pointer to the raw tensor values.
    T *val = nullptr;
    /// Total number of elements in the tensor.
    size_t len = 0;

    /// @brief Default constructor.
    Tensor() {}

    /**
     * @brief Destructor.
     *
     * Releases memory allocated for shape, stride, and value arrays.
     */
    ~Tensor() {
        delete[] shape;
        delete[] stride;
        delete[] val;
    }

    // std::copy constructor
    Tensor(const Tensor<T> &other) {
        nDims = other.nDims;
        shape = new int[nDims];
        std::copy(other.shape, other.shape + nDims, shape);
        stride = new int[nDims];
        std::copy(other.stride, other.stride + nDims, stride);

        len = other.len;
        val = new T[len];
        std::copy(other.val, other.val + len, val);
    }

    /**
     * @brief Copy constructor.
     *
     * Deep copies shape, stride, and value arrays from another tensor.
     *
     * @param other The tensor to copy from.
     */
    Tensor &operator=(const Tensor &other) {
        if (this != &other) {
            if (nDims != other.nDims) {
                nDims = other.nDims;
                delete[] shape;
                shape = new int[nDims];
                delete[] stride;
                stride = new int[nDims];
            }
            std::copy(other.shape, other.shape + nDims, shape);
            std::copy(other.stride, other.stride + nDims, stride);

            if (len != other.len) {
                len = other.len;
                delete[] val;
                val = new T[len];
            }
            std::copy(other.val, other.val + len, val);
        }

        return *this;
    }

    /**
     * @brief Move constructor.
     *
     * Transfers ownership of the data from another tensor.
     *
     * @param other The tensor to move from.
     */
    Tensor(Tensor &&other) noexcept
        : shape(other.shape), stride(other.stride), nDims(other.nDims),
          val(other.val), len(other.len) {
        other.shape = nullptr;
        other.stride = nullptr;
        other.nDims = 0;
        other.val = nullptr;
        other.len = 0;
    }

    /**
     * @brief Move assignment operator.
     *
     * Transfers ownership of the data from another tensor.
     *
     * @param other The tensor to move from.
     * @return Reference to this tensor.
     */
    Tensor &operator=(Tensor &&other) {
        if (this != &other) {
            delete[] shape;
            delete[] stride;
            delete[] val;

            shape = other.shape;
            stride = other.stride;
            nDims = other.nDims;
            val = other.val;
            len = other.len;

            other.shape = nullptr;
            other.stride = nullptr;
            other.nDims = 0;
            other.val = nullptr;
            other.len = 0;
        }

        return *this;
    }

    /**
     * @brief Constructor from shape and value pointer, automatically computes
     * stride.
     *
     * @param shape Shape array.
     * @param nDims Number of dimensions.
     * @param val Value array.
     * @param len Number of elements (must match shape product).
     */
    Tensor(int *shape, size_t nDims, T *val, size_t len)
        : nDims(nDims), len(1) {
        this->shape = shape;
        stride = new int[this->nDims];

        int i = this->nDims - 1;

        stride[i] = 1;
        this->len *= this->shape[i];
        for (i--; i >= 0; i--) {
            stride[i] = this->shape[i + 1] * stride[i + 1];
            this->len *= this->shape[i];
        }
        for (int i = 0; i < this->nDims; i++) {
            stride[i] = this->shape[i] > 1 ? stride[i] : 0;
        }

        if (this->len != len) {
            throw std::invalid_argument("array size suggested by this->shape "
                                        "does not match val argument");
        }

        this->val = val;
    }

    /**
     * @brief Constructor from shape and optional fill value.
     *
     * Stride is automatically computed and tensor is filled with the given
     * value.
     *
     * @param shape Shape array.
     * @param nDims Number of dimensions.
     * @param fill_value Fill value (default 0).
     */
    Tensor(int *shape, size_t nDims, T fill_value = 0) : nDims(nDims), len(1) {
        this->shape = shape;
        stride = new int[this->nDims];

        int i = this->nDims - 1;

        stride[i] = 1;
        len *= this->shape[i];
        for (i--; i >= 0; i--) {
            stride[i] = this->shape[i + 1] * stride[i + 1];
            len *= this->shape[i];
        }
        for (int i = 0; i < this->nDims; i++) {
            stride[i] = this->shape[i] > 1 ? stride[i] : 0;
        }

        val = new T[len];
        std::fill(val, val + len, fill_value);
    }

    /**
     * @brief Creates a view of the tensor.
     *
     * @return A tView<T> structure representing a non-owning view of the
     * tensor.
     */
    struct tView<T> view() const {
        return tView<T>(shape, stride, nDims, val, len);
    }

    /**
     * @brief Stream output operator.
     *
     * Prints the tensor in a readable format.
     *
     * @param stream Output stream.
     * @param tensor The tensor to print.
     * @return Reference to the output stream.
     */
    friend std::ostream &operator<<(std::ostream &stream, Tensor<T> tensor) {
        if (tensor.nDims == 0) {
            std::cout << "[]";
        } else {
            int *cords = new int[tensor.nDims];
            std::fill(cords, cords + tensor.nDims, 0);
            int indent = 0;

            detail::print_tensor(stream, cords, tensor.shape, tensor.stride,
                                 tensor.nDims, tensor.val, 0, indent);

            delete[] cords;
        }
        return stream;
    }
};
} // namespace kaad
