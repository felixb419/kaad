#include "../../include/kaad/tensor/print_tensor.hpp"
#include "../../include/kaad/tensor/tensor.hpp" // for Tensor
#include "kaad/scalar.hpp"                      // for Scalar
#include <cstddef>                              // for size_t
#include <vector>                               // for vector

namespace kaad {

void print_tensor_values(std::ostream &stream, std::span<int> cords,
                         std::span<const int> shape,
                         std::span<const int> stride,
                         std::span<const Scalar> elements, std::size_t idx,
                         std::size_t &indent) {
    const std::size_t rank = shape.size();
    if (idx == rank) {
        int idx = 0;
        for (std::size_t i = 0; i < rank; i++) {
            idx += (cords[i] % shape[i]) * stride[i];
        }
        stream << elements[idx];
    } else {
        int lim = shape[idx];
        stream << "[";
        indent++;
        // iterate for size of current dimension
        for (int i = 0; i < lim - 1; i++) {
            // print next dimension
            print_tensor_values(stream, cords, shape, stride, elements, idx + 1,
                                indent);
            stream << ", ";
            bool indent_here = false;
            for (std::size_t j = 0; j < rank - idx - 1; j++) {
                stream << "\n";
                indent_here = true;
            }
            for (std::size_t j = 0; j < indent && indent_here; j++) {
                stream << " ";
            }
            cords[idx]++;
        }
        // last pass without trailing comma
        print_tensor_values(stream, cords, shape, stride, elements, idx + 1,
                            indent);
        cords[idx]++;

        stream << "]";
        indent--;
    }
}

void print_tensor_impl(std::ostream &stream, std::span<const int> shape,
                       std::span<const int> stride,
                       std::span<const Scalar> elements) {

    stream << "shape: (";
    if (!shape.empty()) {

        stream << shape[0];
        for (Tensor::size_type i = 1; i < shape.size(); i++) {
            stream << ", " << shape[i];
        }
    }
    stream << ")\n";

    stream << "elements: ";

    if (elements.empty()) {
        stream << "[]";
    } else {
        stream << "\n";

        std::vector<int> cords(shape.size());
        std::size_t indent = 0;
        std::size_t idx = 0;

        print_tensor_values(stream, cords, shape, stride, elements, idx,
                            indent);
    }
}

} // namespace kaad
