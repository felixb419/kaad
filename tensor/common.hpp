#pragma once

#include <iostream> // for std::ostream, std::endl
#include <vector>   // for std::vector

namespace kaad::detail {

template <typename T>
inline void print_tensor(std::ostream &stream, std::vector<int> &cords,
                         const std::vector<int> &shape,
                         const std::vector<int> &stride,
                         const std::vector<T> &val, int ind, int &indent) {
    const int nDims = static_cast<int>(shape.size());
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
            print_tensor(stream, cords, shape, stride, val, ind + 1, indent);
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
        print_tensor(stream, cords, shape, stride, val, ind + 1, indent);
        cords[ind]++;

        stream << "]";
        indent--;
    }
}

} // namespace kaad::detail
