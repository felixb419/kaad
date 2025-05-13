#pragma once

#include <stdexcept>
#include <random>
#include <iostream>
#include <initializer_list>
#include <concepts>

using namespace std;

template <typename T>
concept arithmetic = integral<T> || floating_point<T>;
template <typename T>
concept printable = requires(ostream& os, T a) {
    { os << a } -> same_as<ostream&>;
};

inline size_t getIndex(int* indeces, int* shape, int* stride, size_t len) {
    size_t linearIndex = 0;
    for (size_t i = 0; i < len; i++) {
        linearIndex += (indeces[i] % shape[i]) * stride[i];
    }
    return linearIndex;
}

template <printable T>
inline void _print(ostream& stream, int* cords, int* shape, int* stride, size_t shapeLen, T* val, int ind, int& indent) {
    if (ind == shapeLen) {
        stream << val[getIndex(cords, shape, stride, shapeLen)];
    }
    else {
        int lim = shape[ind];
        stream << "[";
        indent++;
        // iterate for size of current dimension
        for (int i = 0; i < lim - 1; i++) {
            // print next dimension
            _print(stream, cords, shape, stride, shapeLen, val, ind + 1, indent);
            stream << ", ";
            bool indent_here = false;
            for (int j = 0; j < shapeLen - ind - 1; j++) {
                stream << endl;
                indent_here = true;
            }
            for (int j = 0; j < indent && indent_here; j++) {
                stream << " ";
            }
            cords[ind]++;
        }
        // last pass without trailing comma
        _print(stream, cords, shape, stride, shapeLen, val, ind + 1, indent);
        cords[ind]++;

        stream << "]";
        indent--;
    }
}

template <arithmetic T>
struct tView {
    int* shape;
    int* stride;
    size_t shapeLen;
    T* val;
    size_t len;

    tView(int* _shape, int* _stride, size_t _shapeLen, T* _val, size_t _len)
    : shape(_shape), stride(_stride), shapeLen(_shapeLen), val(_val), len(_len) {}

    tView(const tView& other) : shape(other.shape), stride(other.stride), shapeLen(other.shapeLen), val(other.val), len(other.len) {}

    tView() : shape(nullptr), stride(nullptr), shapeLen(0), val(nullptr), len(0) {}

    friend ostream& operator<<(ostream& stream, tView<T> view) requires printable<T> {
        if (view.shapeLen == 0) {
            cout << "[]";
        }
        else {
            int* cords = new int[view.shapeLen];
            fill(cords, cords + view.shapeLen, 0);
            int indent = 0;
        
            _print(stream, cords, view.shape, view.stride, view.shapeLen, view.val, 0, indent);
            stream << endl;

            delete[] cords;
        }
        return stream;
    }
};

template<arithmetic T>
class Tensor {
    public:
        int* shape;
        int* stride;
        size_t shapeLen;
        T* val;
        size_t len;

        Tensor() : shape(nullptr), stride(nullptr), shapeLen(0), val(nullptr), len(0) {}

        // destructor
        ~Tensor() {
            delete[] shape;
            delete[] stride;
            delete[] val;
        }

        // copy constructor
        Tensor(const Tensor<T>& other) {
            shapeLen = other.shapeLen;
            shape = new int[shapeLen];
            copy(other.shape, other.shape + shapeLen, shape);
            stride = new int[shapeLen];
            copy(other.stride, other.stride + shapeLen, stride);

            len = other.len;
            val = new T[len];
            copy(other.val, other.val + len, val);
        }
        
        // copy assignment operator
        Tensor& operator=(const Tensor& other) {
            if (this != &other) {
                if (shapeLen != other.shapeLen) {
                    shapeLen = other.shapeLen;
                    delete[] shape;
                    shape = new int[shapeLen];
                    delete[] stride;
                    stride = new int[shapeLen];
                }
                copy(other.shape, other.shape + shapeLen, shape);
                copy(other.stride, other.stride + shapeLen, stride);

                if (len != other.len) {
                    len = other.len;
                    delete[] val;
                    val = new T[len];
                }
                copy(other.val, other.val + len, val);
            }
            
            return *this;
        }
        
        // move constructor
        Tensor(Tensor&& other) noexcept 
        : shape(other.shape), stride(other.stride), shapeLen(other.shapeLen), val(other.val), len(other.len) {
            other.shape = nullptr;
            other.stride = nullptr;
            other.shapeLen = 0;
            other.val = nullptr;
            other.len = 0;
        }

        // move assignment operator
        Tensor& operator=(Tensor&& other) {
            if (this != &other) {
                delete[] shape;
                delete[] stride;
                delete[] val;

                shape = other.shape;
                stride = other.stride;
                shapeLen = other.shapeLen;
                val = other.val;
                len = other.len;

                other.shape = nullptr;
                other.stride = nullptr;
                other.shapeLen = 0;
                other.val = nullptr;
                other.len = 0;
            }

            return *this;
        }

        Tensor(int* _shape, int* _stride, size_t _shapeLen) 
        : shape(_shape), stride(_stride), shapeLen(shapeLen) {
            len = 1;
            for (size_t i = 0; i < shapeLen; i++) {
                len *= shape[i];
            }

            val = new T[len];
            fill(val, val + len, 0);
        }

        Tensor(int* _shape, int* _stride, size_t _shapeLen, T* _val, size_t _len) 
        : shape(_shape), stride(_stride), shapeLen(_shapeLen), val(_val), len(_len) {}

        Tensor(int* _shape, int* _stride, size_t _shapeLen, size_t _len, T _fill=0)
        : shape(_shape), stride(_stride), shapeLen(_shapeLen), len(_len) {
            val = new T[len];
            fill(val, val + len, _fill);
        }

        Tensor(T scalar) : shapeLen(1), len(1) {
            shape = new int[] { 1 };
            stride = new int[] { 0 };
            val = new T[] { scalar };
        }

        Tensor(int* _shape, size_t _shapeLen, T* _val, size_t _len)
        : shapeLen(_shapeLen), len(1) {
            shape = _shape;
            stride = new int[shapeLen];
        
            int i = shapeLen - 1;
        
            stride[i] = 1;
            len *= shape[i];
            for (i--; i >= 0; i--) {
                stride[i] = shape[i+1] * stride[i+1];
                len *= shape[i];
            }
        
            if (len != _len) {
                throw invalid_argument("array size suggested by shape does not match _value argument");
            }

            val = _val;
        }

        Tensor(int* _shape, size_t _shapeLen, T _fill=0)
        : shapeLen(_shapeLen), len(1) {
            shape = _shape;
            stride = new int[shapeLen];
        
            int i = shapeLen - 1;
        
            stride[i] = 1;
            len *= shape[i];
            for (i--; i >= 0; i--) {
                stride[i] = shape[i+1] * stride[i+1];
                len *= shape[i];
            }
        
            val = new T[len];
            fill(val, val + len, _fill);
        }

        struct tView<T> view() const {
            return tView<T>(shape, stride, shapeLen, val, len);
        }

        template <typename... Indeces>
        T operator()(Indeces... indeces) const {
            static_assert(sizeof...(indeces) > 0, "at least one index must be provided");
            static_assert((is_same_v<Indeces, int> && ...), "all indeces must be of type int");
            int len = sizeof...(indeces);
            if (len != shapeLen) {
                throw invalid_argument("indeces do not match Tensor dimensions");
            }
            int cords[len] = {indeces...};
            for (size_t i = 0; i < len; i++) {
                if (cords[i] >= shape[i]) {
                    throw invalid_argument("out of bounds index");
                }
            }
            return val[getIndex(cords, shape, stride, shapeLen)];
        }

        friend ostream& operator<<(ostream& stream, Tensor<T> tensor) {
            if (tensor.shapeLen == 0) {
                cout << "[]";
            }
            else {
                int* cords = new int[tensor.shapeLen];
                fill(cords, cords + tensor.shapeLen, 0);
                int indent = 0;

                _print(stream, cords, tensor.shape, tensor.stride, tensor.shapeLen, tensor.val, 0, indent);
                stream << endl;

                delete[] cords;
            }
            return stream;
        }
};

template <typename T>
void _print_flat(ostream& stream, int* cords, const Tensor<T>& tensor, int ind) {
    if (ind == tensor.shapeLen) {
        stream << tensor.val[getIndex(cords, tensor.shape, tensor.stride, tensor.shapeLen)];
    }
    else {
        stream << "[";
        int lim = tensor.shape[ind];
        // iterate for size of current dimension
        for (int i = 0; i < lim - 1; i++) {
            // print next dimension
            _print_flat(stream, cords, tensor, ind + 1);
            stream << ",";
            cords[ind]++;
        }
        // last pass without trailing comma
        _print_flat(stream, cords, tensor, ind + 1);
        cords[ind]++;

        stream << "]";
    }
}

template <typename T>
void print_flat(const Tensor<T>& tensor, ostream& stream=cout) {
    int* cords = new int[tensor.shapeLen];
    fill(cords, cords + tensor.shapeLen, 0);
    
    _print_flat(stream, cords, tensor, 0);
    
    stream << endl;
    delete[] cords;
}

inline void print_arr(int* arr, size_t len, ostream& stream=cout) {
    stream << "(";
    for (size_t i = 0; i < len; i++) {
        stream << arr[i] << ",";
    }
    stream << ")" << endl;
}
