#pragma once

#include "../include/kaad/tensor/tensor.hpp"           // for Tensor
#include "/home/bert/dev/kaad/include/kaad/scalar.hpp" // for Scalar
#include <algorithm>                                   // for equal
#include <iostream>
#include <span>   // for span
#include <vector> // for vector

inline bool equal_tol(kaad::Scalar a, kaad::Scalar b, kaad::Scalar abs_tol,
                      kaad::Scalar rel_tol) {
    if (a == b) {
        return true;
    }

    const kaad::Scalar diff = std::abs(a - b);
    const kaad::Scalar scale = std::max(std::abs(a), std::abs(b));

    return diff <= abs_tol + rel_tol * scale;
}

inline bool check_tensor(const char *label, const kaad::Tensor &tensor,
                         std::span<const int> shape_correct,
                         std::span<const kaad::Scalar> elements_correct) {
    static kaad::Scalar abs_tol = 1e-6;
    static kaad::Scalar rel_tol = 1e-6;
    if constexpr (std::same_as<kaad::Scalar, double>) {
        abs_tol = 1e-12;
        rel_tol = 1e-12;
    }

    if (!(tensor.rank() == shape_correct.size() &&
          tensor.size() == elements_correct.size())) {
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
        if (!equal_tol(*tensor_it, *elements_it, abs_tol, rel_tol)) {
            std::cerr << "mismatching values in '" << label << "' at idx "
                      << idx << ", act=" << *tensor_it
                      << ", corr=" << *elements_it << std::endl;
            return false;
        }
    }

    return true;
}
