#pragma once

namespace kaad {

/// Enum to signify the mutability of an object.
enum MUTABILITY : char { MUTABLE, IMMUTABLE };

/// Enum to signify special scalar inputs in operations.
enum ScalarOrder : char { NONE_SCALAR, LHS_IS_SCALAR, RHS_IS_SCALAR };

} // namespace kaad
