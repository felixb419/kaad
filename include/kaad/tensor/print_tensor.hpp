#pragma once

#include "../scalar.hpp" // for Scalar
#include <iostream>      // for operator<<, ostream
#include <span>          // for span

namespace kaad {

/**
 * @internal
 */
void print_tensor_impl(std::ostream &stream, std::span<const int> shape,
                       std::span<const int> stride,
                       std::span<const Scalar> elements);

} // namespace kaad
