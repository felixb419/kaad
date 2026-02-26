#include "../include/kaad/kaad.hpp"
#include <algorithm> // for fill
#include <array>     // for array
#include <iostream>  // for operator<<
#include <span>      // for span

int main() {
    kaad::Computation_graph rec;

    std::span<float> a_vals;
    auto a = rec.add_input_node(std::array{3, 2}, a_vals);
    std::fill(a_vals.begin(), a_vals.end(), 10);

    std::span<float> b_vals;
    auto b = rec.add_input_node(std::array{3, 2}, b_vals);
    std::fill(b_vals.begin(), b_vals.end(), 30);

    std::span<float> c_vals;
    auto c = rec.add_input_node(std::array{1}, c_vals);
    std::fill(c_vals.begin(), c_vals.end(), 50);

    std::span<float> d_vals;
    auto d = rec.add_input_node(std::array{2, 3, 1}, d_vals);
    std::fill(d_vals.begin(), d_vals.end(), 20);

    auto ab = add(rec, a, b);
    auto abc = add(rec, ab, c);
    auto res = add(rec, abc, d);

    rec.reset();

    rec.evaluate(std::array{res});

    rec.getGradient(res, std::array{a, b, c, d, res});

    std::cout << "A:\n" << a << std::endl;
    std::cout << "B:\n" << b << std::endl;
    std::cout << "C:\n" << c << std::endl;
    std::cout << "D:\n" << d << std::endl;
    std::cout << "Res:\n" << res << std::endl;

    return 0;
}
