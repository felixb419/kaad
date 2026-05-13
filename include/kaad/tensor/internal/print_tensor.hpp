#pragma once

#include <iostream>
#include <kaad/scalar.hpp>
#include <kaad/tensor/internal/tensor_types.hpp>
#include <span>

namespace kaad {

/// @internal
void print_tensor_impl(std::ostream &stream, ShapeView shape,
                       StridesView strides, std::span<const Scalar> elements);

} // namespace kaad
