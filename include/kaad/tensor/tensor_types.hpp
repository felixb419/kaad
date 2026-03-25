#pragma once

#include <span>
#include <vector>

namespace kaad {

/// Alias for an owning tensor shape.
using Shape = std::vector<int>;
/// Alias for non-owning immutable tensor shape.
using ShapeView = std::span<const int>;

/// Alias for owning tensor strides.
using Stride = std::vector<int>;
/// Alias for non-owning immutable tensor strides.
using Stride_view = std::span<const int>;

} // namespace kaad
