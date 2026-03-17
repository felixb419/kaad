#pragma once

namespace kaad {

#ifdef KAAD_USE_DOUBLE
using Scalar = double;
#else
using Scalar = float;
#endif

} // namespace kaad
