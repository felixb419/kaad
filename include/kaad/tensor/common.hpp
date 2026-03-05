#pragma once

#include "../scalar.hpp" // for Scalar
#include <cstddef>       // for size_t
#include <iostream>      // for operator<<, ostream, endl
#include <vector>        // for vector

namespace kaad::detail {

inline void print_tensor(std::ostream &os, std::vector<int> &cords,
                         const int *shape, const int *stride, std::size_t rank,
                         const Scalar *elements, std::size_t nElems,
                         std::size_t ind, int &indent) {
    if (ind == rank) {
        int idx = 0;
        for (std::size_t i = 0; i < rank; i++) {
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
            print_tensor(os, cords, shape, stride, rank, elements, nElems,
                         ind + 1, indent);
            os << ", ";
            bool indent_here = false;
            for (std::size_t j = 0; j < rank - ind - 1; j++) {
                os << std::endl;
                indent_here = true;
            }
            for (int j = 0; j < indent && indent_here; j++) {
                os << " ";
            }
            cords[ind]++;
        }
        // last pass without trailing comma
        print_tensor(os, cords, shape, stride, rank, elements, nElems, ind + 1,
                     indent);
        cords[ind]++;

        os << "]";
        indent--;
    }
}

} // namespace kaad::detail
