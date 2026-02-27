#pragma once

#include "../include/kaad/tensor/tensor.hpp"           // for Tensor
#include "/home/bert/dev/kaad/include/kaad/scalar.hpp" // for Scalar
#include <algorithm>                                   // for equal
#include <span>                                        // for span
#include <vector>                                      // for vector

inline bool check_tensor(const kaad::Tensor &tensor, std::span<const int> shape,
                         std::span<const kaad::Scalar> elements) {

    if (!(tensor.rank() == shape.size() && tensor.size() == elements.size())) {
        return false;
    }

    if (!(std::equal(shape.begin(), shape.end(), tensor.shape().data()) &&
          std::equal(elements.begin(), elements.end(), tensor.begin()))) {
        return false;
    }

    return true;
}
