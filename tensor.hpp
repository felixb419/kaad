#pragma once

#include <algorithm>        // for std::copy, std::max, std::fill
#include <cstddef>          // for size_t
#include <initializer_list> // for std::initializer_list
#include <iostream>  // for std::operator<<, std::ostream, std::cout, std::end
#include <stdexcept> // for std::invalid_argument

namespace kaad {

/**
 * @brief Recursively prints the contents of a tensor in nested list format.
 *
 * This function is used to format and output the tensor data in a
 * human-readable nested list representation.
 *
 * @tparam T The data type of the tensor values.
 * @param stream The output stream to write the tensor representation to.
 * @param cords The current coordinates (indices) within the tensor.
 * @param shape The shape of the tensor (size of each dimension).
 * @param stride The stride values for indexing into the flat data array.
 * @param nDims The number of dimensions in the tensor.
 * @param val Pointer to the flat array containing the tensor data.
 * @param ind The current recursion depth (dimension index).
 * @param indent Tracks the current indentation level for formatted output.
 */
template <typename T>
inline void print(std::ostream &stream, int *cords, int *shape, int *stride,
                  size_t nDims, T *val, int ind, int &indent) {
    if (ind == nDims) {
        int idx = 0;
        for (int i = 0; i < nDims; i++) {
            idx += (cords[i] % shape[i]) * stride[i];
        }
        stream << val[idx];
    } else {
        int lim = shape[ind];
        stream << "[";
        indent++;
        // iterate for size of current dimension
        for (int i = 0; i < lim - 1; i++) {
            // print next dimension
            print(stream, cords, shape, stride, nDims, val, ind + 1, indent);
            stream << ", ";
            bool indent_here = false;
            for (int j = 0; j < nDims - ind - 1; j++) {
                stream << std::endl;
                indent_here = true;
            }
            for (int j = 0; j < indent && indent_here; j++) {
                stream << " ";
            }
            cords[ind]++;
        }
        // last pass without trailing comma
        print(stream, cords, shape, stride, nDims, val, ind + 1, indent);
        cords[ind]++;

        stream << "]";
        indent--;
    }
}

/**
 * @brief Lightweight view into a tensor's shape, stride, and data.
 *
 * This struct allows for non-owning access to tensor metadata and values,
 * useful for read-only operations or efficient slicing without copying.
 *
 * @tparam T The data type stored in the tensor.
 */
template <typename T> struct tView {
    /// Pointer to the shape array, where shape[i] is the size along dimension
    /// i.
    int *shape = nullptr;
    /// Pointer to the stride array, where stride[i] gives the step to the next
    /// element in dimension i.
    int *stride = nullptr;
    /// Number of dimensions in the tensor.
    size_t nDims = 0;
    /// Pointer to the flat data array storing the tensor elements.
    T *val = nullptr;
    /// Total number of elements in the tensor.
    size_t len = 0;

    /**
     * @brief Default constructor.
     */
    tView() {}

    /**
     * @brief Copy constructor.
     */
    tView(const tView &other)
        : shape(other.shape), stride(other.stride), nDims(other.nDims),
          val(other.val), len(other.len) {}

    /**
     * @brief Constructs a tView from shape, stride, and data pointer.
     *
     * @param shape Pointer to the shape array.
     * @param stride Pointer to the stride array.
     * @param nDims Number of dimensions.
     * @param val Pointer to the tensor values.
     * @param len Total number of elements.
     */
    tView(int *shape, int *stride, size_t nDims, T *val, size_t len)
        : shape(shape), stride(stride), nDims(nDims), val(val), len(len) {}

    /**
     * @brief Overloads the stream output operator to print the tensor view.
     *
     * @param stream Output stream.
     * @param view The tensor view to print.
     * @return std::ostream& The updated output stream.
     */
    friend std::ostream &operator<<(std::ostream &stream, tView<T> view) {
        if (view.nDims == 0) {
            std::cout << "[]";
        } else {
            int *cords = new int[view.nDims];
            std::fill(cords, cords + view.nDims, 0);
            int indent = 0;

            print(stream, cords, view.shape, view.stride, view.nDims, view.val,
                  0, indent);

            delete[] cords;
        }
        return stream;
    }
};

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
     * @brief Constructor from shape and stride.
     *
     * Initializes tensor memory based on shape and stride arrays.
     *
     * @param shape Pointer to the shape array.
     * @param stride Pointer to the stride array.
     * @param nDims Number of dimensions.
     */
    Tensor(int *shape, int *stride, size_t nDims)
        : shape(shape), stride(stride), nDims(nDims) {
        len = 1;
        for (size_t i = 0; i < this->nDims; i++) {
            len *= this->shape[i];
        }

        val = new T[len];
        std::fill(val, val + len, 0);
    }

    /**
     * @brief Constructor from shape, stride, value array, and total length.
     *
     * @param shape Shape array.
     * @param stride Stride array.
     * @param nDims Number of dimensions.
     * @param val Value array.
     * @param len Total number of elements.
     */
    Tensor(int *shape, int *stride, size_t nDims, T *val, size_t len)
        : shape(shape), stride(stride), nDims(nDims), val(val), len(len) {}

    /**
     * @brief Constructor from shape, stride, length, and a fill value.
     *
     * Allocates a new tensor and fills it with the given value.
     *
     * @param shape Shape array.
     * @param stride Stride array.
     * @param nDims Number of dimensions.
     * @param len Number of elements.
     * @param fill_value Value to fill the tensor with (default 0).
     */
    Tensor(int *shape, int *stride, size_t nDims, size_t len, T fill_value = 0)
        : shape(shape), stride(stride), nDims(nDims), len(len) {
        val = new T[this->len];
        std::fill(val, val + this->len, fill_value);
    }

    /**
     * @brief Constructor for scalar tensors.
     *
     * Creates a 1D tensor with a single element.
     *
     * @param scalar The scalar value.
     */
    Tensor(T scalar) : nDims(1), len(1) {
        shape = new int[]{1};
        stride = new int[]{0};
        val = new T[]{scalar};
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
     * @brief Indexing operator using coordinate list.
     *
     * Throws if coordinates are out of bounds or wrong dimension count.
     *
     * @param cords The coordinate indices.
     * @return The value at the given coordinates.
     */
    const T &operator()(std::initializer_list<int> cords) {
        const int *begin = cords.begin();
        const int *end = cords.end();
        if (end - begin != nDims) {
            throw std::invalid_argument(
                "incorrect number of coordinate dimension");
        }
        int index = 0;
        for (const int *c_p = begin, *sh_p = shape, *st_p = stride; c_p != end;
             c_p++) {
            if (*c_p >= *sh_p || *c_p < 0) {
                throw std::invalid_argument("out of bound coordinate");
            }
            index += (*c_p) * (*st_p);
        }
        return shape[index];
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

            print(stream, cords, tensor.shape, tensor.stride, tensor.nDims,
                  tensor.val, 0, indent);

            delete[] cords;
        }
        return stream;
    }
};
} // namespace kaad
