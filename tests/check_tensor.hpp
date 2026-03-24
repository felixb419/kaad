#pragma once

#include "../include/kaad/scalar.hpp"        // for Scalar
#include "../include/kaad/tensor/tensor.hpp" // for Tensor
#include <algorithm>                         // for equal
#include <iostream>
#include <span>   // for span
#include <vector> // for vector

inline bool equal_tol(kaad::Scalar lhs, kaad::Scalar rhs, kaad::Scalar abs_tol,
                      kaad::Scalar rel_tol) {
    if (lhs == rhs) {
        return true;
    }

    kaad::Scalar diff = std::abs(lhs - rhs);
    kaad::Scalar scale = std::max(std::abs(lhs), std::abs(rhs));

    return diff <= abs_tol + (rel_tol * scale);
}

inline bool check_tensor(const char *label, const kaad::Tensor &tensor,
                         std::span<const int> shape_correct,
                         std::span<const kaad::Scalar> elements_correct) {
    constexpr bool SCALAR_IS_DOUBLE = std::same_as<kaad::Scalar, double>;
    constexpr kaad::Scalar ABS_TOL = SCALAR_IS_DOUBLE ? 1e-12 : 1e-6;
    constexpr kaad::Scalar REL_TOL = SCALAR_IS_DOUBLE ? 1e-12 : 1e-6;

    if (tensor.rank() != shape_correct.size() ||
        tensor.size() != elements_correct.size()) {
        return false;
    }

    if (!std::equal(shape_correct.begin(), shape_correct.end(),
                    tensor.shape().data())) {
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
