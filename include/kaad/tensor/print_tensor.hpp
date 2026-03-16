#pragma once

#include <iostream>        // for operator<<, ostream
#include <kaad/scalar.hpp> // for Scalar
#include <span>            // for span

namespace kaad {

/**
 * @internal
 */
void print_tensor_impl(std::ostream &stream, std::span<const int> shape,
                       std::span<const int> stride,
                       std::span<const Scalar> elements);

} // namespace kaad
