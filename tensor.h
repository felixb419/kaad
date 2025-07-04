#pragma once

#include <stddef.h>          // for size_t
#include <algorithm>         // for copy, max, fill
#include <initializer_list>  // for initializer_list
#include <iostream>          // for operator<<, ostream, cout, endl, basic_o...
#include <stdexcept>         // for invalid_argument

namespace kaad {
    inline size_t getIndex(int* indeces, int* shape, int* stride, size_t len) {
        size_t linearIndex = 0;
        for (size_t i = 0; i < len; i++) {
            linearIndex += (indeces[i] % shape[i]) * stride[i];
        }
        return linearIndex;
    }

    template <typename T>
    inline void _print(std::ostream& stream, int* cords, int* shape, int* stride, size_t nDims, T* val, int ind, int& indent) {
        if (ind == nDims) {
            stream << val[getIndex(cords, shape, stride, nDims)];
        }
        else {
            int lim = shape[ind];
            stream << "[";
            indent++;
            // iterate for size of current dimension
            for (int i = 0; i < lim - 1; i++) {
                // print next dimension
                _print(stream, cords, shape, stride, nDims, val, ind + 1, indent);
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
            _print(stream, cords, shape, stride, nDims, val, ind + 1, indent);
            cords[ind]++;

            stream << "]";
            indent--;
        }
    }

    template <typename T>
    struct tView {
        int* shape = nullptr;
        int* stride = nullptr;
        size_t nDims = 0;
        T* val = nullptr;
        size_t len = 0;

        tView(int* _shape, int* _stride, size_t _nDims, T* _val, size_t _len)
        : shape(_shape), stride(_stride), nDims(_nDims), val(_val), len(_len) {}

        tView(const tView& other) : shape(other.shape), stride(other.stride), nDims(other.nDims), val(other.val), len(other.len) {}

        tView() {}

        friend std::ostream& operator<<(std::ostream& stream, tView<T> view) {
            if (view.nDims == 0) {
                std::cout << "[]";
            }
            else {
                int* cords = new int[view.nDims];
                std::fill(cords, cords + view.nDims, 0);
                int indent = 0;
            
                _print(stream, cords, view.shape, view.stride, view.nDims, view.val, 0, indent);

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
            size_t nDims = 0;
            T* val = nullptr;
            size_t len = 0;

            Tensor() {}

            // destructor
            ~Tensor() {
                delete[] shape;
                delete[] stride;
                delete[] val;
            }

            // std::copy constructor
            Tensor(const Tensor<T>& other) {
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
            Tensor& operator=(const Tensor& other) {
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
            Tensor(Tensor&& other) noexcept 
            : shape(other.shape), stride(other.stride), nDims(other.nDims), val(other.val), len(other.len) {
                other.shape = nullptr;
                other.stride = nullptr;
                other.nDims = 0;
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

            Tensor(int* _shape, int* _stride, size_t _nDims) 
            : shape(_shape), stride(_stride), nDims(_nDims) {
                len = 1;
                for (size_t i = 0; i < nDims; i++) {
                    len *= shape[i];
                }

                val = new T[len];
                std::fill(val, val + len, 0);
            }

            Tensor(int* _shape, int* _stride, size_t _nDims, T* _val, size_t _len) 
            : shape(_shape), stride(_stride), nDims(_nDims), val(_val), len(_len) {}

            Tensor(int* _shape, int* _stride, size_t _nDims, size_t _len, T _fill=0)
            : shape(_shape), stride(_stride), nDims(_nDims), len(_len) {
                val = new T[len];
                std::fill(val, val + len, _fill);
            }

            Tensor(T scalar) : nDims(1), len(1) {
                shape = new int[] { 1 };
                stride = new int[] { 0 };
                val = new T[] { scalar };
            }

            Tensor(int* _shape, size_t _nDims, T* _val, size_t _len)
            : nDims(_nDims), len(1) {
                shape = _shape;
                stride = new int[nDims];
            
                int i = nDims - 1;
            
                stride[i] = 1;
                len *= shape[i];
                for (i--; i >= 0; i--) {
                    stride[i] = shape[i+1] * stride[i+1];
                    len *= shape[i];
                }
                for (int i = 0; i < nDims; i++) {
                    stride[i] = shape[i] > 1 ? stride[i] : 0;
                }
            
                if (len != _len) {
                    throw std::invalid_argument("array size suggested by shape does not match _value argument");
                }

                val = _val;
            }

            Tensor(int* _shape, size_t _nDims, T _fill=0)
            : nDims(_nDims), len(1) {
                shape = _shape;
                stride = new int[nDims];
            
                int i = nDims - 1;
            
                stride[i] = 1;
                len *= shape[i];
                for (i--; i >= 0; i--) {
                    stride[i] = shape[i+1] * stride[i+1];
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

            const T& operator()(std::initializer_list<int> cords) {
                const int* begin = cords.begin();
                const int* end = cords.end();
                if (end - begin != nDims) {
                    throw std::invalid_argument("incorrect number of coordinate dimension");
                }
                int index = 0;
                for (const int* c_p = begin, *sh_p = shape, *st_p = stride; c_p != end; c_p++) {
                    if (*c_p >= *sh_p || *c_p < 0) {
                        throw std::invalid_argument("out of bound coordinate");
                    }
                    index += (*c_p) * (*st_p);
                }
                return shape[index];
            }

            friend std::ostream& operator<<(std::ostream& stream, Tensor<T> tensor) {
                if (tensor.nDims == 0) {
                    std::cout << "[]";
                }
                else {
                    int* cords = new int[tensor.nDims];
                    std::fill(cords, cords + tensor.nDims, 0);
                    int indent = 0;

                    _print(stream, cords, tensor.shape, tensor.stride, tensor.nDims, tensor.val, 0, indent);

                    delete[] cords;
                }
                return stream;
            }
    };

    template <typename T>
    void _print_flat(std::ostream& os, int* cords, const Tensor<T>& tensor, int ind) {
        if (ind == tensor.nDims) {
            os << tensor.val[getIndex(cords, tensor.shape, tensor.stride, tensor.nDims)];
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
    void print_flat(const Tensor<T>& tensor, std::ostream& stream=std::cout) {
        int* cords = new int[tensor.nDims];
        std::fill(cords, cords + tensor.nDims, 0);
        
        _print_flat(stream, cords, tensor, 0);
        
        stream << std::endl;
        delete[] cords;
    }

    inline void print_arr(const int* begin, const int* end, std::ostream& os=std::cout) {
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
    bool combine_flexible(int* shape1, const size_t nDims1, int* shape2, const size_t nDims2, int* newShape, size_t newLen) {
        int ind = newLen - 1;
        for (int i = 1; i <= newLen; i++, ind--) {
            int ind1 = nDims1 - i;
            int ind2 = nDims2 - i;
            if (ind1 >= 0 && ind2 >= 0) {
                if (shape1[ind1] != shape2[ind2] && shape1[ind1] != 1 && shape2[ind2] != 1) {
                    return false;
                }
                newShape[ind] = std::max(shape1[ind1], shape2[ind2]);
            }
            else {
                newShape[ind] = ind1 >= 0 ? shape1[ind1] : shape2[ind2];
            }
        }
        return true;
    }

    // returns a dynamically allocated array that represents the resulting shape of broadcasting two tensors by matrix multiplication
    // matmul: (n?,k),(k,m?) -> (n?,m?)
    bool combine_matrix(int* shape1, const size_t nDims1, int* shape2, const size_t nDims2, int* newShape, size_t newLen) {
        if (shape1[nDims1 - 1] != shape2[nDims2 - 2]) {
            return false;
        }
        std::fill(newShape, newShape + newLen, 0);

        newShape[newLen - 1] = shape2[nDims2 - 1];
        newShape[newLen - 2] = shape1[nDims1 - 2];

        int ind = newLen - 3;
        for (int i = 3; i <= newLen; i++, ind--) {
            int ind1 = nDims1 - i;
            int ind2 = nDims2 - i;
            if (ind1 >= 0 && ind2 >= 0) {
                if (shape1[ind1] != shape2[ind2] && shape1[ind1] != 1 && shape2[ind2] != 1) {
                    return false;
                }
                newShape[ind] = std::max(shape1[ind1], shape2[ind2]);
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
        std::copy(shape, shape + len - 2, shape_T);
        shape_T[len - 2] = shape[len - 1];
        shape_T[len - 1] = shape[len - 2];

        std::copy(stride, stride + len - 2, stride_T);
        stride_T[len - 2] = stride[len - 1];
        stride_T[len - 1] = stride[len - 2];
    }
}    
