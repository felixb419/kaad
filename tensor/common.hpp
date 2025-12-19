#pragma once

#include <cstddef>  // for std::size_t
#include <iostream> // for std::ostream, std::endl

namespace kaad::detail {

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
inline void print_tensor(std::ostream &stream, int *cords, int *shape,
                         int *stride, std::size_t nDims, T *val, int ind,
                         int &indent) {
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
            print_tensor(stream, cords, shape, stride, nDims, val, ind + 1,
                         indent);
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
        print_tensor(stream, cords, shape, stride, nDims, val, ind + 1, indent);
        cords[ind]++;

        stream << "]";
        indent--;
    }
}

} // namespace kaad::detail
