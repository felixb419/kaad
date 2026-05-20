#pragma once

#include "kaad/tensor/internal/tensor.hpp"
#include "kaad/tensor/internal/tensor_types.hpp"

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <iostream>
#include <kaad/scalar.hpp>
#include <kaad/tensor/tensor_view.hpp>
#include <span>

inline bool equal_tol(kaad::Scalar lhs, kaad::Scalar rhs, kaad::Scalar abs_tol,
                      kaad::Scalar rel_tol) {
    if (lhs == rhs) {
        return true;
    }

    kaad::Scalar diff = std::abs(lhs - rhs);
    kaad::Scalar scale = std::max(std::abs(lhs), std::abs(rhs));

    return diff <= abs_tol + (rel_tol * scale);
}

inline bool check_tensor(const char *label, kaad::TensorView tensor,
                         kaad::ShapeView shape_correct,
                         std::span<const kaad::Scalar> elements_correct) {
    constexpr bool SCALAR_IS_DOUBLE = std::same_as<kaad::Scalar, double>;
    constexpr kaad::Scalar ABS_TOL = SCALAR_IS_DOUBLE ? 1e-12 : 1e-6;
    constexpr kaad::Scalar REL_TOL = SCALAR_IS_DOUBLE ? 1e-12 : 1e-6;

    if (tensor.rank() != shape_correct.size() ||
        tensor.size() != elements_correct.size()) {
        return false;
    }

    if (!std::ranges::equal(shape_correct, tensor.shape())) {
        return false;
    }

    if (!std::ranges::equal(shape_correct, tensor.shape())) {
        return false;
    }

    int idx = 0;
    auto tensor_it = tensor.begin();
    auto elements_it = elements_correct.begin();

    for (; elements_it != elements_correct.end();
         tensor_it++, elements_it++, idx++) {
        if (!equal_tol(*tensor_it, *elements_it, ABS_TOL, REL_TOL)) {
            std::cerr << "mismatching values in '" << label << "' at idx "
                      << idx << ", act=" << *tensor_it
                      << ", corr=" << *elements_it << '\n';
            return false;
        }
    }

    return true;
}
