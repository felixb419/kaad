#pragma once

#include "../operations/operation_concept.hpp"

namespace kaad::operations {

template <class Policy>
concept ReductionPolicy = HasOpName<Policy>;

struct SumPolicy {

    static constexpr const char *OPERATION_NAME = "sum";
};

struct MeanPolicy {

    static constexpr const char *OPERATION_NAME = "mean";
};

} // namespace kaad::operations
