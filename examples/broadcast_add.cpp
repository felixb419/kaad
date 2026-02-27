#include "../include/kaad/kaad.hpp"
#include <algorithm> // for fill
#include <array>     // for array
#include <iostream>  // for operator<<
#include <span>      // for span

int main() {
    // Create computation graph.
    kaad::Computation_graph rec;

    // Add input nodes to the graph.
    std::span<float> a_vals; // span to represent the element array of a
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

    // Add computation nodes to graph via operators.
    auto ab = add(rec, a, b);    // [3,2] + [3,2] -> [3,2]
    auto abc = add(rec, ab, c);  // [3,2] + [1] -> [3,2]
    auto res = add(rec, abc, d); // [3,2] + [2,3,1] -> [2,3,2]

    // Reset the graph.
    rec.reset();

    // Evaluate 'res'.
    rec.evaluate(std::array{res});

    // Compute the gradient of res w.r.t. to a, b, c and d.
    rec.getGradient(res, std::array{a, b, c, d});

    // Print values of nodes.
    std::cout << "A:\n" << a << std::endl;
    std::cout << "B:\n" << b << std::endl;
    std::cout << "C:\n" << c << std::endl;
    std::cout << "D:\n" << d << std::endl;
    std::cout << "Res:\n" << res << std::endl;

    return 0;
}
