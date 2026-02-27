#include "../include/kaad/kaad.hpp"
#include <algorithm> // for fill
#include <array>     // for array
#include <iostream>  // for operator<<
#include <span>      // for span

int main() {
    // Create computation graph.
    kaad::Computation_graph rec;

    // Add input nodes to the graph.
    std::span<float> a_vals; // Add input nodes to the graph
    auto a = rec.add_input_node(std::array{5}, a_vals);
    std::fill(a_vals.begin(), a_vals.end(), 10);

    std::span<float> b_vals;
    auto b = rec.add_input_node(std::array{5}, b_vals);
    std::fill(b_vals.begin(), b_vals.end(), 50);

    std::span<float> c_vals;
    auto c = rec.add_input_node(std::array{5}, c_vals);
    std::fill(c_vals.begin(), c_vals.end(), 20);

    // Add computation nodes to graph via operators.
    kaad::Node_handle ab = dot(rec, a, b);   // [5] * [5] -> [1]
    kaad::Node_handle res = dot(rec, ab, c); // [1] * [5] -> [1]

    // Reset the graph.
    rec.reset();

    // Evaluate 'res'.
    rec.evaluate(std::array{res});

    // Compute the gradient of res w.r.t. to a, b and c.
    rec.getGradient(res, std::array{a, b, c});

    // Print values of nodes.
    std::cout << "A:\n" << a << std::endl;
    std::cout << "B:\n" << b << std::endl;
    std::cout << "C:\n" << c << std::endl;
    std::cout << "Res:\n" << res << std::endl;

    return 0;
}
