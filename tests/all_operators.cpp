#include "../include/kaad/kaad.hpp"
#include <array> // for array
#include <span>  // for span

void call_all_operators(kaad::Computation_graph &rec, kaad::Node_handle a,
                        kaad::Node_handle b) {
    add(rec, a, b);
    dot(rec, a, b);
    matmul(rec, a, b);
    outer(rec, a, b);

    exp(rec, a);
    transpose(rec, a, {5, 4, 3, 2, 1});
    sum(rec, a);
    sum(rec, a, 1);
    mean(rec, a);
    mean(rec, a, 1);
    slice(rec, a, {1, 1, 1}, {3, 3, 3});
}

int main() {
    kaad::Computation_graph rec;

    std::span<kaad::Scalar> vals; // dummy value in this test
    kaad::Node_handle a = rec.add_input_node(std::array{8, 8}, vals);
    kaad::Node_handle b = rec.add_input_node(std::array{4, 5}, vals);
    kaad::Node_handle d = rec.add_input_node(std::array{4, 5}, vals);

    call_all_operators(rec, a, b);

    return 0;
}
