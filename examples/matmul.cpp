#include "../include/kaad/kaad.hpp"
#include <algorithm> // for fill
#include <array>     // for array
#include <iostream>  // for operator<<
#include <span>      // for span

int main() {
    kaad::Computation_graph rec;

    std::span<float> a_vals;
    auto a = rec.add_input_node(std::array{3, 5}, a_vals);
    std::fill(a_vals.begin(), a_vals.end(), 10);

    std::span<float> b_vals;
    auto b = rec.add_input_node(std::array{5, 8}, b_vals);
    std::fill(b_vals.begin(), b_vals.end(), 50);

    std::span<float> c_vals;
    auto c = rec.add_input_node(std::array{2, 2, 8, 2}, c_vals);
    std::fill(c_vals.begin(), c_vals.end(), 20);

    kaad::Node_handle ab = matmul(rec, a, b);
    kaad::Node_handle res = matmul(rec, ab, c);

    rec.reset();

    auto e = rec.evaluate(std::array{res});

    auto g = rec.getGradient(res, std::array{a, b, c});

    std::cout << "A:\n" << a << std::endl;
    std::cout << "B:\n" << b << std::endl;
    std::cout << "C:\n" << c << std::endl;
    std::cout << "Res:\n" << res << std::endl;

    return 0;
}
