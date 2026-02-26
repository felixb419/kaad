#include "../include/kaad/kaad.hpp"
#include <array>    // for array
#include <iostream> // for operator<<
#include <numeric>  // for iota
#include <span>     // for span

int main() {
    kaad::Computation_graph rec;

    std::span<float> a_vals;
    kaad::Node_handle a = rec.add_input_node(std::array{3, 5, 2}, a_vals);
    std::iota(a_vals.begin(), a_vals.end(), 0);

    kaad::Node_handle c = transpose(rec, a);

    rec.reset();

    auto e = rec.evaluate(std::array{c});

    auto g = rec.getGradient(c, a);

    std::cout << "A:\n" << a << std::endl;
    std::cout << "C:\n" << c << std::endl;

    return 0;
}
