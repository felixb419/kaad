#include "../kaad.hpp"
#include "../scalar.hpp"
#include <iostream>

void call_all_operators(kaad::Computation_graph &rec, kaad::Node_handle a,
                        kaad::Node_handle b) {
    add<kaad::Scalar>(rec, a, b);
    dot<kaad::Scalar>(rec, a, b);
    matmul<kaad::Scalar>(rec, a, b);
    outer<kaad::Scalar>(rec, a, b);

    exp<kaad::Scalar>(rec, a);
    transpose<kaad::Scalar>(rec, a, {5, 4, 3, 2, 1});
    sum<kaad::Scalar>(rec, a);
    sum<kaad::Scalar>(rec, a, 1);
    mean<kaad::Scalar>(rec, a);
    mean<kaad::Scalar>(rec, a, 1);
    slice<kaad::Scalar>(rec, a, {1, 1, 1}, {3, 3, 3});
}

int main() {
    kaad::Computation_graph rec;

    kaad::Node_handle a = rec.append(std::vector<int>{8, 8}, 10);
    kaad::Node_handle b = rec.append(std::vector<int>{4, 5}, 50);
    kaad::Node_handle d = rec.append(std::vector<int>{4, 5}, 20);

    call_all_operators(rec, a, b);

    return 0;
}
