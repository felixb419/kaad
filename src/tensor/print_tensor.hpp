#pragma once

#include <iostream>                     // for ostream
#include <kaad/scalar.hpp>              // for Scalar
#include <kaad/tensor/tensor_types.hpp> // for Shape_view, Stride_view
#include <span>                         // for span

namespace kaad {

/// @internal
void print_tensor_impl(std::ostream &stream, Shape_view shape,
                       Stride_view stride, std::span<const Scalar> elements);

} // namespace kaad
