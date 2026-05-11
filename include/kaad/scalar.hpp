#pragma once

namespace kaad {

#ifdef KAAD_USE_DOUBLE
/// @brief Type alias for a basic scalar value.
using Scalar = double;
#else
/// @brief Type alias for a basic scalar value.
using Scalar = float;
#endif

} // namespace kaad
