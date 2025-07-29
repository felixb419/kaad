#pragma once

#include "tensor.h"
#include <iostream>

namespace kaad {

template <typename T>
void print_flat_impl(std::ostream &os, int *cords, const Tensor<T> &tensor,
                     int ind) {
  if (ind == tensor.nDims) {
    os << tensor
              .val[getIndex(cords, tensor.shape, tensor.stride, tensor.nDims)];
  } else {
    os << "[";
    int lim = tensor.shape[ind];
    // iterate for size of current dimension
    for (int i = 0; i < lim - 1; i++) {
      // print next dimension
      print_flat_impl(os, cords, tensor, ind + 1);
      os << ",";
      cords[ind]++;
    }
    // last pass without trailing comma
    print_flat_impl(os, cords, tensor, ind + 1);
    cords[ind]++;

    os << "]";
  }
}

template <typename T>
void print_flat(const Tensor<T> &tensor, std::ostream &stream = std::cout) {
  int *cords = new int[tensor.nDims];
  std::fill(cords, cords + tensor.nDims, 0);

  print_flat_impl(stream, cords, tensor, 0);

  stream << std::endl;
  delete[] cords;
}

template <typename T>
inline void print_arr(const T *begin, const T *end,
                      std::ostream &os = std::cout) {
  os << "[";
  for (const T *p = begin; p != end; p++) {
    if (p != begin) {
      os << ",";
    }
    os << *p;
  }
  os << "]";
}

// returns a dynamically allocated array that represents the resulting shape of
// broadcasting two tensors d1_n == d2_n || d1_n == 1 || d2_n == 1
bool combine_flexible(int *shape1, const size_t nDims1, int *shape2,
                      const size_t nDims2, int *newShape, size_t newLen) {
  int ind = newLen - 1;
  for (int i = 1; i <= newLen; i++, ind--) {
    int ind1 = nDims1 - i;
    int ind2 = nDims2 - i;
    if (ind1 >= 0 && ind2 >= 0) {
      if (shape1[ind1] != shape2[ind2] && shape1[ind1] != 1 &&
          shape2[ind2] != 1) {
        return false;
      }
      newShape[ind] = std::max(shape1[ind1], shape2[ind2]);
    } else {
      newShape[ind] = ind1 >= 0 ? shape1[ind1] : shape2[ind2];
    }
  }
  return true;
}

// returns a dynamically allocated array that represents the resulting shape of
// broadcasting two tensors by matrix multiplication matmul: (n?,k),(k,m?) ->
// (n?,m?)
bool combine_matrix(int *shape1, const size_t nDims1, int *shape2,
                    const size_t nDims2, int *newShape, size_t newLen) {
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
      if (shape1[ind1] != shape2[ind2] && shape1[ind1] != 1 &&
          shape2[ind2] != 1) {
        return false;
      }
      newShape[ind] = std::max(shape1[ind1], shape2[ind2]);
    } else {
      newShape[ind] = ind1 >= 0 ? shape1[ind1] : shape2[ind2];
    }
  }
  return true;
}

void transp(int *shape, int *stride, size_t len) {
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

void transp(int *shape, int *stride, size_t len, int *shape_T, int *stride_T) {
  for (int i = 0, j = len - 1; i < len; i++, j--) {
    shape_T[j] = shape[i];
    stride_T[j] = stride[i];
  }
}

void transp2D(int *shape, int *stride, size_t len) {
  int temp;
  temp = shape[len - 2];
  shape[len - 2] = shape[len - 1];
  shape[len - 1] = temp;

  temp = stride[len - 2];
  stride[len - 2] = stride[len - 1];
  stride[len - 1] = temp;
}

void transp2D(int *shape, int *stride, size_t len, int *shape_T,
              int *stride_T) {
  std::copy(shape, shape + len - 2, shape_T);
  shape_T[len - 2] = shape[len - 1];
  shape_T[len - 1] = shape[len - 2];

  std::copy(stride, stride + len - 2, stride_T);
  stride_T[len - 2] = stride[len - 1];
  stride_T[len - 1] = stride[len - 2];
}
} // namespace kaad
