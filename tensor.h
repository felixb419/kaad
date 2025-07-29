#pragma once

#include <algorithm>        // for copy, max, fill
#include <initializer_list> // for initializer_list
#include <iostream>         // for operator<<, ostream, cout, endl, basic_o...
#include <stddef.h>         // for size_t
#include <stdexcept>        // for invalid_argument

namespace kaad {
inline size_t getIndex(int *indeces, int *shape, int *stride, size_t len) {
    size_t linearIndex = 0;
    for (size_t i = 0; i < len; i++) {
        linearIndex += (indeces[i] % shape[i]) * stride[i];
    }
    return linearIndex;
}

template <typename T>
inline void print(std::ostream &stream, int *cords, int *shape, int *stride,
                  size_t nDims, T *val, int ind, int &indent) {
    if (ind == nDims) {
        stream << val[getIndex(cords, shape, stride, nDims)];
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

template <typename T> struct tView {
    int *shape = nullptr;
    int *stride = nullptr;
    size_t nDims = 0;
    T *val = nullptr;
    size_t len = 0;

    tView(int *_shape, int *_stride, size_t _nDims, T *_val, size_t _len)
        : shape(_shape), stride(_stride), nDims(_nDims), val(_val), len(_len) {}

    tView(const tView &other)
        : shape(other.shape), stride(other.stride), nDims(other.nDims),
          val(other.val), len(other.len) {}

    tView() {}

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

template <typename T> class Tensor {
  public:
    int *shape = nullptr;
    int *stride = nullptr;
    size_t nDims = 0;
    T *val = nullptr;
    size_t len = 0;

    Tensor() {}

    // destructor
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

    // std::copy assignment operator
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

    // move constructor
    Tensor(Tensor &&other) noexcept
        : shape(other.shape), stride(other.stride), nDims(other.nDims),
          val(other.val), len(other.len) {
        other.shape = nullptr;
        other.stride = nullptr;
        other.nDims = 0;
        other.val = nullptr;
        other.len = 0;
    }

    // move assignment operator
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

    Tensor(int *_shape, int *_stride, size_t _nDims)
        : shape(_shape), stride(_stride), nDims(_nDims) {
        len = 1;
        for (size_t i = 0; i < nDims; i++) {
            len *= shape[i];
        }

        val = new T[len];
        std::fill(val, val + len, 0);
    }

    Tensor(int *_shape, int *_stride, size_t _nDims, T *_val, size_t _len)
        : shape(_shape), stride(_stride), nDims(_nDims), val(_val), len(_len) {}

    Tensor(int *_shape, int *_stride, size_t _nDims, size_t _len, T _fill = 0)
        : shape(_shape), stride(_stride), nDims(_nDims), len(_len) {
        val = new T[len];
        std::fill(val, val + len, _fill);
    }

    Tensor(T scalar) : nDims(1), len(1) {
        shape = new int[]{1};
        stride = new int[]{0};
        val = new T[]{scalar};
    }

    Tensor(int *_shape, size_t _nDims, T *_val, size_t _len)
        : nDims(_nDims), len(1) {
        shape = _shape;
        stride = new int[nDims];

        int i = nDims - 1;

        stride[i] = 1;
        len *= shape[i];
        for (i--; i >= 0; i--) {
            stride[i] = shape[i + 1] * stride[i + 1];
            len *= shape[i];
        }
        for (int i = 0; i < nDims; i++) {
            stride[i] = shape[i] > 1 ? stride[i] : 0;
        }

        if (len != _len) {
            throw std::invalid_argument(
                "array size suggested by shape does not match _value argument");
        }

        val = _val;
    }

    Tensor(int *_shape, size_t _nDims, T _fill = 0) : nDims(_nDims), len(1) {
        shape = _shape;
        stride = new int[nDims];

        int i = nDims - 1;

        stride[i] = 1;
        len *= shape[i];
        for (i--; i >= 0; i--) {
            stride[i] = shape[i + 1] * stride[i + 1];
            len *= shape[i];
        }
        for (int i = 0; i < nDims; i++) {
            stride[i] = shape[i] > 1 ? stride[i] : 0;
        }

        val = new T[len];
        std::fill(val, val + len, _fill);
    }

    struct tView<T> view() const {
        return tView<T>(shape, stride, nDims, val, len);
    }

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
