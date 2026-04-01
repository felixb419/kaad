#include "print_tensor.hpp"

#include <cstddef>                      // for size_t
#include <kaad/scalar.hpp>              // for Scalar
#include <kaad/static_vector.hpp>       // for StaticVector
#include <kaad/tensor/tensor.hpp>       // for Tensor
#include <kaad/tensor/tensor_types.hpp> // for ShapeView, StridesView

namespace kaad {

void print_tensor_values(std::ostream &stream, std::span<int> cords,
                         ShapeView shape, StridesView strides,
                         std::span<const Scalar> elements, std::size_t idx,
                         std::size_t &indent) {
    std::size_t rank = shape.size();
    if (idx == rank) {
        std::size_t idx = 0;
        for (std::size_t i = 0; i < rank; i++) {
            idx += (cords[i] % shape[i]) * strides[i];
        }
        stream << elements[idx];
    } else {
        extent lim = shape[idx];
        stream << "[";
        indent++;
        // iterate for size of current dimension
        for (std::size_t i = 0; i < lim - 1; i++) {
            // print next dimension
            print_tensor_values(stream, cords, shape, strides, elements,
                                idx + 1, indent);
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
        print_tensor_values(stream, cords, shape, strides, elements, idx + 1,
                            indent);
        cords[idx]++;

        stream << "]";
        indent--;
    }
}

void print_tensor_impl(std::ostream &stream, ShapeView shape,
                       StridesView strides, std::span<const Scalar> elements) {

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

        StaticVector<int> cords(shape.size());
        std::size_t indent = 0;
        std::size_t idx = 0;

        print_tensor_values(stream, cords, shape, strides, elements, idx,
                            indent);
    }
}

} // namespace kaad
