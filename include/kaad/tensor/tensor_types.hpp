#pragma once

#include <kaad/max_rank.hpp>      // for KAAD_MAX_RANK
#include <kaad/static_vector.hpp> // for StaticVector

namespace kaad {

/// Alias for the size of a shape along a singular tensor dimension.
using extent = std::size_t;

/// Alias for an owning tensor shape.
using Shape = StaticVector<extent>;

/// Empty Shape object (scalar).
static constexpr Shape SCALAR_SHAPE{};

/// Alias for non-owning immutable tensor shape.
using ShapeView = Shape::view_type;

/// Alias for a singular stride along a singular tensor dimension.
using stride = std::size_t;

/// Alias for owning tensor strides.
using Strides = StaticVector<stride>;

/// Alias for non-owning immutable tensor strides.
using StridesView = Strides::view_type;

} // namespace kaad
