#pragma once

#include <kaad/graph/operation_concept.hpp>

namespace kaad::functions {

template <class Policy>
concept reduction_policy = HasOpName<Policy>;

struct SumPolicy {

    static constexpr const char *OPERATION_NAME = "sum";
};

struct MeanPolicy {

    static constexpr const char *OPERATION_NAME = "mean";
};

} // namespace kaad::functions
