#pragma once

#include <kaad/max_rank.hpp>      // for KAAD_MAX_RANK
#include <kaad/static_vector.hpp> // for StaticVector

namespace kaad {

/// Alias for the size of a shape along a singular tensor dimension.
using extent = int;

/// Alias for an owning tensor shape.
using Shape = StaticVector<extent>;

/// Alias for non-owning immutable tensor shape.
using ShapeView = Shape::view_type;

/// Alias for owning tensor strides.
using Stride = StaticVector<extent>;

/// Alias for non-owning immutable tensor strides.
using StrideView = Stride::view_type;

} // namespace kaad
