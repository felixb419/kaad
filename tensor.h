#pragma once

#include <stdexcept>
#include <random>
#include <iostream>

using namespace std;

inline size_t getIndex(int* indeces, int* shape, int* stride, size_t len) {
    size_t linearIndex = 0;
    for (size_t i = 0; i < len; i++) {
        linearIndex += (indeces[i] % shape[i]) * stride[i];
    }
    return linearIndex;
}

template <typename T>
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

template <typename T>
struct tView {
    int* shape = nullptr;
    int* stride = nullptr;
    size_t shapeLen = 0;
    T* val = nullptr;
    size_t len = 0;

    tView(int* _shape, int* _stride, size_t _shapeLen, T* _val, size_t _len)
    : shape(_shape), stride(_stride), shapeLen(_shapeLen), val(_val), len(_len) {}

    tView(const tView& other) : shape(other.shape), stride(other.stride), shapeLen(other.shapeLen), val(other.val), len(other.len) {}

    tView() {}

    friend ostream& operator<<(ostream& stream, tView<T> view) {
        if (view.shapeLen == 0) {
            cout << "[]";
        }
        else {
            int* cords = new int[view.shapeLen];
            fill(cords, cords + view.shapeLen, 0);
            int indent = 0;
        
            _print(stream, cords, view.shape, view.stride, view.shapeLen, view.val, 0, indent);

            delete[] cords;
        }
        return stream;
    }
};

template<typename T>
class Tensor {
    public:
        int* shape = nullptr;
        int* stride = nullptr;
        size_t shapeLen = 0;
        T* val = nullptr;
        size_t len = 0;

        Tensor() {}

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
        : shape(_shape), stride(_stride), shapeLen(_shapeLen) {
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
            for (int i = 0; i < shapeLen; i++) {
                stride[i] = shape[i] > 1 ? stride[i] : 0;
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
            for (int i = 0; i < shapeLen; i++) {
                stride[i] = shape[i] > 1 ? stride[i] : 0;
            }
        
            val = new T[len];
            fill(val, val + len, _fill);
        }

        struct tView<T> view() const {
            return tView<T>(shape, stride, shapeLen, val, len);
        }

        const T& operator()(initializer_list<int> cords) {
            int* begin = cords.begin();
            int* end = cords.end();
            if (end - begin != shapeLen) {
                throw invalid_argument("incorrect number of coordinate dimension");
            }
            int index = 0;
            for (int* c_p = begin, *sh_p = shape, *st_p = stride; c_p != end; c_p++) {
                if (*c_p >= *sh_p || *c_p < 0) {
                    throw invalid_argument("out of bound coordinate");
                }
                index += (*c_p) * (*st_p);
            }
            return shape[index];
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

                delete[] cords;
            }
            return stream;
        }
};

template <typename T>
void _print_flat(ostream& os, int* cords, const Tensor<T>& tensor, int ind) {
    if (ind == tensor.shapeLen) {
        os << tensor.val[getIndex(cords, tensor.shape, tensor.stride, tensor.shapeLen)];
    }
    else {
        os << "[";
        int lim = tensor.shape[ind];
        // iterate for size of current dimension
        for (int i = 0; i < lim - 1; i++) {
            // print next dimension
            _print_flat(os, cords, tensor, ind + 1);
            os << ",";
            cords[ind]++;
        }
        // last pass without trailing comma
        _print_flat(os, cords, tensor, ind + 1);
        cords[ind]++;

        os << "]";
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

inline void print_arr(const int* begin, const int* end, ostream& os=cout) {
    os << "[";
    for (const int* p = begin; p != end; p++) {
        if (p != begin) {
            os << ",";
        }
        os << *p;
    }
    os << "]";
}

// returns a dynamically allocated array that represents the resulting shape of broadcasting two tensors
// d1_n == d2_n || d1_n == 1 || d2_n == 1
bool combine_flexible(int* shape1, const size_t shapeLen1, int* shape2, const size_t shapeLen2, int* newShape, size_t newLen) {
    int ind = newLen - 1;
    for (int i = 1; i <= newLen; i++, ind--) {
        int ind1 = shapeLen1 - i;
        int ind2 = shapeLen2 - i;
        if (ind1 >= 0 && ind2 >= 0) {
            if (shape1[ind1] != shape2[ind2] && shape1[ind1] != 1 && shape2[ind2] != 1) {
                return false;
            }
            newShape[ind] = max(shape1[ind1], shape2[ind2]);
        }
        else {
            newShape[ind] = ind1 >= 0 ? shape1[ind1] : shape2[ind2];
        }
    }
    return true;
}

// returns a dynamically allocated array that represents the resulting shape of broadcasting two tensors by matrix multiplication
// matmul: (n?,k),(k,m?) -> (n?,m?)
bool combine_matrix(int* shape1, const size_t shapeLen1, int* shape2, const size_t shapeLen2, int* newShape, size_t newLen) {
    if (shape1[shapeLen1 - 1] != shape2[shapeLen2 - 2]) {
        return false;
    }
    fill(newShape, newShape + newLen, 0);

    newShape[newLen - 1] = shape2[shapeLen2 - 1];
    newShape[newLen - 2] = shape1[shapeLen1 - 2];

    int ind = newLen - 3;
    for (int i = 3; i <= newLen; i++, ind--) {
        int ind1 = shapeLen1 - i;
        int ind2 = shapeLen2 - i;
        if (ind1 >= 0 && ind2 >= 0) {
            if (shape1[ind1] != shape2[ind2] && shape1[ind1] != 1 && shape2[ind2] != 1) {
                return false;
            }
            newShape[ind] = max(shape1[ind1], shape2[ind2]);
        }
        else {
            newShape[ind] = ind1 >= 0 ? shape1[ind1] : shape2[ind2];
        }
    }
    return true;
}

void transp(int* shape, int* stride, size_t len) {
    int temp;
    for (int i = 0, j = len - 1; i < len / 2; i++, j--) {
        temp = shape[i];
        shape[i] = shape[j];
        shape[j] = temp;

        temp = stride[i];
        stride[i] = stride[j];
        stride[j] = temp;
    }
}

void transp(int* shape, int* stride,size_t len, int* shape_T, int* stride_T) {
    for (int i = 0, j = len - 1; i < len; i++, j--) {
        shape_T[j] = shape[i];
        stride_T[j] = stride[i];
    }
}

void transp2D(int* shape, int* stride, size_t len) {
    int temp;
    temp = shape[len - 2];
    shape[len - 2] = shape[len - 1];
    shape[len - 1] = temp;

    temp = stride[len - 2];
    stride[len - 2] = stride[len - 1];
    stride[len - 1] = temp;
}

void transp2D(int* shape, int* stride,size_t len, int* shape_T, int* stride_T) {
    copy(shape, shape + len - 2, shape_T);
    shape_T[len - 2] = shape[len - 1];
    shape_T[len - 1] = shape[len - 2];

    copy(stride, stride + len - 2, stride_T);
    stride_T[len - 2] = stride[len - 1];
    stride_T[len - 1] = stride[len - 2];
}
