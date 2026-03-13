#pragma once

#include "../scalar.hpp" // for Scalar
#include <cstddef>       // for size_t
#include <iostream>      // for operator<<, ostream
#include <vector>        // for vector

namespace kaad {

void print_tensor(std::ostream &stream, std::vector<int> &cords,
                  const int *shape, const int *stride, std::size_t rank,
                  const Scalar *elements, std::size_t nElems, std::size_t ind,
                  int &indent);

} // namespace kaad
