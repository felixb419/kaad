#pragma once

#include <iostream>                     // for ostream
#include <kaad/scalar.hpp>              // for Scalar
#include <kaad/tensor/tensor_types.hpp> // for ShapeView, StridesView
#include <span>                         // for span

namespace kaad {

/// @internal
void print_tensor_impl(std::ostream &stream, ShapeView shape,
                       StridesView strides, std::span<const Scalar> elements);

} // namespace kaad
