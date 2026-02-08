#pragma once

#include "../scalar.hpp" // for Scalar
#include <iostream>      // for std::ostream, std::endl
#include <vector>        // for std::vector

namespace kaad::detail {

inline void print_tensor(std::ostream &os, std::vector<int> &cords,
                         const int *shape, const int *stride, size_t nDims,
                         const Scalar *elements, size_t nElems, int ind,
                         int &indent) {
    if (ind == nDims) {
        int idx = 0;
        for (int i = 0; i < nDims; i++) {
            idx += (cords[i] % shape[i]) * stride[i];
        }
        os << elements[idx];
    } else {
        int lim = shape[ind];
        os << "[";
        indent++;
        // iterate for size of current dimension
        for (int i = 0; i < lim - 1; i++) {
            // print next dimension
            print_tensor(os, cords, shape, stride, nDims, elements, nElems,
                         ind + 1, indent);
            os << ", ";
            bool indent_here = false;
            for (int j = 0; j < nDims - ind - 1; j++) {
                os << std::endl;
                indent_here = true;
            }
            for (int j = 0; j < indent && indent_here; j++) {
                os << " ";
            }
            cords[ind]++;
        }
        // last pass without trailing comma
        print_tensor(os, cords, shape, stride, nDims, elements, nElems, ind + 1,
                     indent);
        cords[ind]++;

        os << "]";
        indent--;
    }
}

} // namespace kaad::detail
