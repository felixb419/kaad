#include "../../include/kaad/tensor/common.hpp"
#include "kaad/scalar.hpp" // for Scalar

namespace kaad {

void print_tensor(std::ostream &stream, std::vector<int> &cords,
                  const int *shape, const int *stride, std::size_t rank,
                  const Scalar *elements, std::size_t nElems, std::size_t ind,
                  int &indent) {
    if (ind == rank) {
        int idx = 0;
        for (std::size_t i = 0; i < rank; i++) {
            idx += (cords[i] % shape[i]) * stride[i];
        }
        stream << elements[idx];
    } else {
        int lim = shape[ind];
        stream << "[";
        indent++;
        // iterate for size of current dimension
        for (int i = 0; i < lim - 1; i++) {
            // print next dimension
            print_tensor(stream, cords, shape, stride, rank, elements, nElems,
                         ind + 1, indent);
            stream << ", ";
            bool indent_here = false;
            for (std::size_t j = 0; j < rank - ind - 1; j++) {
                stream << "\n";
                indent_here = true;
            }
            for (int j = 0; j < indent && indent_here; j++) {
                stream << " ";
            }
            cords[ind]++;
        }
        // last pass without trailing comma
        print_tensor(stream, cords, shape, stride, rank, elements, nElems,
                     ind + 1, indent);
        cords[ind]++;

        stream << "]";
        indent--;
    }
}

} // namespace kaad
